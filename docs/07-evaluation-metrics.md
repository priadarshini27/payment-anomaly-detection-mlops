# 07 — Evaluation Metrics

[← 06 — Modeling](06-modeling.md) | [Index](README.md) | Next: [08 — Drift Monitoring →](08-drift-monitoring.md)

## Why accuracy is excluded

Because fraud prevalence is under 0.2%, accuracy is not reported as a
headline metric anywhere in this project — a model predicting "not fraud"
for every transaction would score above 99.8% accuracy while catching
zero fraud. Evaluation instead centers on metrics that remain meaningful
under severe class imbalance:

- **Precision–Recall AUC (PR-AUC)** — the primary metric. More
  informative than ROC-AUC when positives are rare, since it does not
  credit the model for the (trivially large) true-negative rate.
- **ROC-AUC** — reported alongside PR-AUC for comparability with
  published PaySim benchmarks.
- **Precision@k / Recall@k** — evaluated at k = the top 0.5% of scored
  transactions, matching realistic manual-review capacity: in production
  only a limited number of flagged transactions can actually be reviewed
  per day.
- **F1 at the operating threshold** — the anomaly-score cutoff is chosen
  from precision@k/recall@k rather than defaulting to sklearn's raw
  `decision_function` sign.
- **Confusion matrix, stratified by transaction type** — surfaces whether
  errors concentrate in TRANSFER/CASH_OUT (expected, since that's where
  all real fraud lives per [02-dataset-and-eda.md](02-dataset-and-eda.md))
  or elsewhere (a warning sign of a mis-specified feature).

All metrics are persisted to `model_runs` alongside the training
timestamp and hyperparameters, so successive model versions can be
compared over time directly from SQL rather than only from ad-hoc
notebook output.

## `models/evaluate.py`

### File: models/evaluate.py

```python
"""
models/evaluate.py
=====================
Evaluates a trained Isolation Forest against the held-out isFraud label
(used here, ONLY here, strictly for offline evaluation -- never as a model
feature, preserving the unsupervised setting per Document 1 Section 1/5).

Because fraud prevalence is under 0.2%, accuracy is deliberately excluded
as a headline metric (Document 1, Section 5): a model that always predicts
"not fraud" would score >99.8% accuracy while catching zero fraud.
Evaluation instead centers on:
  PR-AUC            -- primary metric; more informative than ROC-AUC when
                        positives are rare.
  ROC-AUC            -- reported alongside PR-AUC for comparability with
                        published PaySim benchmarks.
  precision@k/recall@k -- k = top 0.5% of scored transactions by anomaly
                        score, matching realistic manual-review capacity.
  F1 at threshold      -- the anomaly-score cutoff is chosen to balance
                        precision/recall, not sklearn's raw decision sign.
  confusion matrix by type -- surfaces whether errors concentrate in
                        TRANSFER/CASH_OUT (expected) or elsewhere (a
                        warning sign of a mis-specified feature).

All metrics are written back to the model_runs row for the evaluated run,
so successive model versions can be compared directly from SQL.

Run directly:
    python models/evaluate.py
Evaluates the latest trained (or --run-id specified) model against the
full `transactions` table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.predict import load_model_and_scaler, score  # noqa: E402
from sql.queries import get_engine, DEFAULT_DB_PATH  # noqa: E402

K_FRACTION = 0.005  # top 0.5% of scored transactions, per Document 1 Section 5


def get_model_run(engine, run_id: int | None):
    with engine.connect() as conn:
        if run_id is not None:
            row = conn.execute(
                text("SELECT * FROM model_runs WHERE run_id = :rid"), {"rid": run_id}
            ).mappings().fetchone()
        else:
            row = conn.execute(
                text("SELECT * FROM model_runs ORDER BY run_id DESC LIMIT 1")
            ).mappings().fetchone()
    if row is None:
        raise RuntimeError("No model_runs row found. Run models/train.py first.")
    return dict(row)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k_fraction: float) -> tuple[float, float, float]:
    """
    Returns (precision@k, recall@k, threshold) where k = the top k_fraction
    of transactions by anomaly score are treated as flagged.
    """
    n = len(scores)
    k = max(1, int(round(n * k_fraction)))
    order = np.argsort(-scores)  # descending by score
    top_k_idx = order[:k]
    threshold = scores[order[k - 1]]

    y_pred = np.zeros(n, dtype=int)
    y_pred[top_k_idx] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, float(threshold)


def evaluate(db_path: str | None = None, run_id: int | None = None, k_fraction: float = K_FRACTION,
             read_chunksize: int = 200_000) -> dict:
    engine = get_engine(db_path or DEFAULT_DB_PATH)
    run = get_model_run(engine, run_id)

    model, scaler = load_model_and_scaler(run["model_path"], run["scaler_path"])

    # Read and score in chunks rather than pd.read_sql("SELECT * ...") in one
    # shot -- at PaySim's full 6.36M-row scale the unchunked read builds a
    # large intermediate Python-object array before typed conversion, which
    # can exhaust memory (see models/train.py::load_training_data for the
    # same issue hit during training). PR-AUC/ROC-AUC/confusion-matrix
    # computation need every row's score at once, so the scored chunks are
    # still concatenated afterward -- this only caps the *read* peak, not
    # the final in-memory size, but that's the part that was failing.
    scored_chunks = [
        score(chunk, model, scaler)
        for chunk in pd.read_sql(text("SELECT * FROM transactions;"), engine, chunksize=read_chunksize)
    ]
    scored = pd.concat(scored_chunks, ignore_index=True)

    y_true = scored["isFraud"].to_numpy()
    scores = scored["anomaly_score"].to_numpy()

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    precision_at_k, recall_at_k, threshold = precision_recall_at_k(y_true, scores, k_fraction)

    y_pred_at_threshold = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred_at_threshold, zero_division=0)

    cm_overall = confusion_matrix(y_true, y_pred_at_threshold)

    cm_by_type = {}
    for t in scored["type"].unique():
        mask = scored["type"] == t
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            continue
        cm_by_type[t] = confusion_matrix(y_true[mask], y_pred_at_threshold[mask]).tolist()

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "f1_at_threshold": float(f1),
        "threshold": float(threshold),
        "k_fraction": float(k_fraction),
    }

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE model_runs
                SET pr_auc = :pr_auc, roc_auc = :roc_auc,
                    precision_at_k = :precision_at_k, recall_at_k = :recall_at_k,
                    f1_at_threshold = :f1_at_threshold, threshold = :threshold,
                    k_fraction = :k_fraction, is_deployed = 1
                WHERE run_id = :run_id;
                """
            ),
            {**metrics, "run_id": run["run_id"]},
        )
        # Only one deployed model at a time -- undeploy any prior run.
        conn.execute(
            text("UPDATE model_runs SET is_deployed = 0 WHERE run_id != :run_id;"),
            {"run_id": run["run_id"]},
        )

    print("=" * 60)
    print(f"Evaluation for model_runs.run_id = {run['run_id']}")
    print("=" * 60)
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Precision@{k_fraction:.1%}:      {precision_at_k:.4f}")
    print(f"Recall@{k_fraction:.1%}:         {recall_at_k:.4f}")
    print(f"F1 @ threshold:       {f1:.4f}")
    print(f"Threshold (score):    {threshold:.4f}")
    print(f"Confusion matrix (overall, @ threshold):\n{cm_overall}")
    print("Confusion matrix by type:")
    for t, cm in cm_by_type.items():
        print(f"  {t}: {cm}")
    print("=" * 60)

    return {"run_id": run["run_id"], **metrics, "confusion_matrix_overall": cm_overall.tolist(),
            "confusion_matrix_by_type": cm_by_type}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--k-fraction", type=float, default=K_FRACTION)
    args = parser.parse_args()
    evaluate(db_path=args.db_path, run_id=args.run_id, k_fraction=args.k_fraction)


if __name__ == "__main__":
    main()
```

## Running it

```bash
python models/evaluate.py
```

Evaluates the most recently trained model (or a specific one via
`--run-id`) against the full `transactions` table's `isFraud` label,
writes the metrics back into that model's `model_runs` row, and marks it
`is_deployed = 1` (undeploying any prior version).

### Measured result (full 6,362,604-row dataset, model_runs.run_id = 1)

```
============================================================
Evaluation for model_runs.run_id = 1
============================================================
PR-AUC:               0.0114
ROC-AUC:              0.8540
Precision@0.5%:      0.0257
Recall@0.5%:         0.0997
F1 @ threshold:       0.0407
Threshold (score):    -0.0340
Confusion matrix (overall, @ threshold):
[[6323256   31151]
 [   7380     817]]
Confusion matrix by type:
  TRANSFER: [[502019, 26793], [3283, 814]]
  CASH_OUT: [[2233093, 291], [4097, 3]]
============================================================

real    6m8.644s
```

Reading these numbers against the base rate rather than in isolation, per
[02-dataset-and-eda.md](02-dataset-and-eda.md)'s guidance that accuracy is
meaningless here: the base fraud rate is 0.129%
(`8,197 / 6,362,604`), so a PR-AUC of 0.0114 represents roughly an **8.8x
enrichment** over random guessing — the model concentrates real fraud far
above its base rate in its highest-scored transactions, even though the
absolute PR-AUC looks small in isolation (expected under this severity of
class imbalance; see Document 1 Section 5). ROC-AUC of 0.854 confirms
solid overall rank-ordering. Note that no `nameOrig`-based train/test
split was performed for this run — the model was evaluated against the
same full dataset it was trained on (consistent with the project's design
of using `isFraud` only for offline evaluation, never for train/test
partitioning of an unsupervised model) — so these numbers describe
in-sample discriminative power, not held-out generalization; a rigorous
production benchmark would score a genuinely unseen window instead (e.g.
the last N `step`s, mirroring how `drift/retrain.py` treats a recent
window as "current").

As documented in [02-dataset-and-eda.md](02-dataset-and-eda.md), the
confusion matrix by type confirms errors concentrate almost entirely in
`TRANSFER` and `CASH_OUT` — exactly the two types where real fraud lives
in PaySim — rather than spreading across `CASH_IN`/`DEBIT`/`PAYMENT`,
which would have been a warning sign of a mis-specified feature.

## Reading `model_runs` from SQL

```sql
SELECT run_id, trained_at, pr_auc, roc_auc, precision_at_k, recall_at_k,
       f1_at_threshold, is_deployed
FROM model_runs
ORDER BY run_id DESC;
```

This is the query the dashboard's Model Performance view
(`sql/queries.py::get_model_run_history`) runs to build its PR-AUC/ROC-AUC
trend chart across model versions.

---
[← 06 — Modeling](06-modeling.md) | [Index](README.md) | Next: [08 — Drift Monitoring →](08-drift-monitoring.md)
