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
- **Train vs held-out gap** — every metric above is computed on data the
  model was not trained on, and the training window is scored too, so
  overfitting shows up as a gap (see the generalization check below).

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

Held-out evaluation:
  Headline metrics are computed only on the held-out window (step >
  model_runs.train_step_end) that models/train.py never fit on. The
  training window is also scored, and the train-vs-test gap is reported
  so overfitting to the training period is visible rather than hidden
  inside an in-sample number.

Run directly:
    python models/evaluate.py
Evaluates the latest trained (or --run-id specified) model on its held-out
window (or, for legacy runs with no recorded split, on the full table with
an in-sample warning).
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
from models.train import ensure_split_columns  # noqa: E402
from preprocessing.feature_engineering import MODEL_FEATURE_COLUMNS  # noqa: E402
from sql.queries import get_engine, DEFAULT_DB_PATH  # noqa: E402

K_FRACTION = 0.005  # top 0.5% of scored transactions, per Document 1 Section 5

# Overfitting flag: training-window ROC-AUC exceeding held-out ROC-AUC by
# more than this.
OVERFIT_ROC_AUC_GAP = 0.05


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


def score_window(engine, model, scaler, where: str = "", params: dict | None = None,
                 read_chunksize: int = 200_000) -> pd.DataFrame:
    """
    Scores every transaction matching `where` (an SQL WHERE clause, or "" for
    all rows).

    Reads and scores in chunks rather than pd.read_sql("SELECT * ...") in one
    shot -- at PaySim's full 6.36M-row scale the unchunked read builds a
    large intermediate Python-object array before typed conversion, which
    can exhaust memory (see models/train.py::load_training_data for the
    same issue hit during training). PR-AUC/ROC-AUC/confusion-matrix
    computation need every row's score at once, so the scored chunks are
    still concatenated afterward -- this only caps the *read* peak, not
    the final in-memory size, but that's the part that was failing.
    """
    cols = "txn_id, step, type, isFraud, " + ", ".join(MODEL_FEATURE_COLUMNS)
    sql = text(f"SELECT {cols} FROM transactions {where};")
    scored_chunks = [
        score(chunk, model, scaler)
        for chunk in pd.read_sql(sql, engine, params=params or {}, chunksize=read_chunksize)
    ]
    return pd.concat(scored_chunks, ignore_index=True)


def generalization_gap(train_y: np.ndarray, train_scores: np.ndarray,
                       test_y: np.ndarray, test_scores: np.ndarray) -> dict:
    """
    Compares in-sample (training window) vs held-out ranking quality.

    ROC-AUC is insensitive to the fraud base rate, so its train-test gap is
    what decides the overfitting flag. PR-AUC is *not*: PaySim's base rate
    is ~4.4x higher after the default split than before it, which moves
    both raw PR-AUC and lift (PR-AUC / base rate, capped at 1 / base rate)
    without any change in ranking quality. Both are reported for context
    only.
    """
    train_pr_auc = average_precision_score(train_y, train_scores)
    train_roc_auc = roc_auc_score(train_y, train_scores)
    test_pr_auc = average_precision_score(test_y, test_scores)
    test_roc_auc = roc_auc_score(test_y, test_scores)
    train_lift = train_pr_auc / train_y.mean()
    test_lift = test_pr_auc / test_y.mean()
    roc_auc_gap = train_roc_auc - test_roc_auc
    return {
        "train_pr_auc": float(train_pr_auc),
        "train_roc_auc": float(train_roc_auc),
        "train_pr_auc_lift": float(train_lift),
        "test_pr_auc_lift": float(test_lift),
        "roc_auc_gap": float(roc_auc_gap),
        "overfit_warning": bool(roc_auc_gap > OVERFIT_ROC_AUC_GAP),
    }


def evaluate(db_path: str | None = None, run_id: int | None = None, k_fraction: float = K_FRACTION,
             read_chunksize: int = 200_000) -> dict:
    engine = get_engine(db_path or DEFAULT_DB_PATH)
    ensure_split_columns(engine)
    run = get_model_run(engine, run_id)

    model, scaler = load_model_and_scaler(run["model_path"], run["scaler_path"])

    train_step_end = run.get("train_step_end")
    gap = None
    if train_step_end is None:
        # Legacy run trained on every row -- there is no held-out data, so
        # these metrics are in-sample only.
        print("WARNING: this run has no train_step_end (trained on the full table). "
              "Metrics below are IN-SAMPLE and overstate real-world performance. "
              "Retrain with models/train.py to get a held-out evaluation.")
        scored = score_window(engine, model, scaler, read_chunksize=read_chunksize)
    else:
        scored = score_window(engine, model, scaler, "WHERE step > :s", {"s": train_step_end},
                              read_chunksize=read_chunksize)
        if scored.empty or scored["isFraud"].sum() == 0:
            raise RuntimeError(f"Held-out window (step > {train_step_end}) has no rows or no fraud to evaluate.")
        train_scored = score_window(engine, model, scaler, "WHERE step <= :s", {"s": train_step_end},
                                    read_chunksize=read_chunksize)
        gap = generalization_gap(
            train_scored["isFraud"].to_numpy(), train_scored["anomaly_score"].to_numpy(),
            scored["isFraud"].to_numpy(), scored["anomaly_score"].to_numpy(),
        )
        del train_scored

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
                    k_fraction = :k_fraction,
                    train_pr_auc = :train_pr_auc, train_roc_auc = :train_roc_auc,
                    is_deployed = 1
                WHERE run_id = :run_id;
                """
            ),
            {
                **metrics,
                "train_pr_auc": gap["train_pr_auc"] if gap else None,
                "train_roc_auc": gap["train_roc_auc"] if gap else None,
                "run_id": run["run_id"],
            },
        )
        # Only one deployed model at a time -- undeploy any prior run.
        conn.execute(
            text("UPDATE model_runs SET is_deployed = 0 WHERE run_id != :run_id;"),
            {"run_id": run["run_id"]},
        )

    print("=" * 60)
    print(f"Evaluation for model_runs.run_id = {run['run_id']}")
    if train_step_end is not None:
        print(f"Held-out window: step > {train_step_end} ({len(scored):,} rows, "
              f"{int(y_true.sum()):,} fraud, base rate {y_true.mean():.4%})")
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
    if gap is not None:
        print("-" * 60)
        print("Generalization check (training window vs held-out window)")
        print(f"ROC-AUC         train {gap['train_roc_auc']:.4f}  test {roc_auc:.4f}  "
              f"gap {gap['roc_auc_gap']:+.4f}")
        print(f"PR-AUC          train {gap['train_pr_auc']:.4f}  test {pr_auc:.4f}  "
              "(base rates differ -- context only)")
        print(f"PR-AUC lift     train {gap['train_pr_auc_lift']:.2f}x  test {gap['test_pr_auc_lift']:.2f}x  "
              "(PR-AUC / base rate)")
        if gap["overfit_warning"]:
            print("WARNING: held-out performance is materially worse than in-sample -- "
                  "the model is not generalizing to later transactions.")
        else:
            print("OK: no material train/test gap.")
    print("=" * 60)

    return {"run_id": run["run_id"], **metrics, "confusion_matrix_overall": cm_overall.tolist(),
            "confusion_matrix_by_type": cm_by_type, "generalization": gap}


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
python models/evaluate.py              # latest model
python models/evaluate.py --run-id 1   # a specific version
```

Evaluates the most recently trained model (or a specific one via
`--run-id`) against the `isFraud` label in its **held-out window**: the
rows with `step > model_runs.train_step_end`, which training never saw
(see [06-modeling.md](06-modeling.md#traintest-split)). It then:

1. computes PR-AUC, ROC-AUC, precision/recall@k, F1 and the per-type
   confusion matrices on the held-out rows;
2. scores the training rows as well and compares the two (the
   **generalization check**, below);
3. writes the held-out metrics to `pr_auc`, `roc_auc`, … and the
   training-window metrics to `train_pr_auc`/`train_roc_auc` in that
   model's `model_runs` row;
4. marks it `is_deployed = 1`, undeploying any prior version.

Runs trained before the split existed (no `train_step_end`) are evaluated
on the full table, with a warning that the numbers are in-sample.

## Generalization check (overfitting)

Comparing the training window with the held-out window is how this
project detects overfitting: a model that has memorized its training
period ranks training rows much better than later ones.

`evaluate.py` decides this with **ROC-AUC only**. If training ROC-AUC
exceeds held-out ROC-AUC by more than `OVERFIT_ROC_AUC_GAP` (0.05), it
prints a warning; otherwise it prints `OK: no material train/test gap.`

PR-AUC can't be used for this comparison. Its random-guess baseline
equals the fraud rate, and PaySim's fraud rate is about 4.4× higher in the
held-out window (0.339%) than in training (0.077%), because transaction
volume collapses in the late steps while fraud keeps arriving. Raw PR-AUC
therefore rises in the held-out window even if the model ranks no better.
Dividing by the base rate ("lift") doesn't fully fix this either, because
lift is capped at 1 / base rate, so a higher fraud rate lowers the
ceiling. ROC-AUC doesn't depend on the base rate, so a gap in ROC-AUC is
a real difference in ranking quality. PR-AUC and lift are still printed
for context.

## Measured result (run 2, time-based split)

```
============================================================
Evaluation for model_runs.run_id = 2
Held-out window: step > 355 (1,248,723 rows, 4,237 fraud, base rate 0.3393%)
============================================================
PR-AUC:               0.0280
ROC-AUC:              0.8538
Precision@0.5%:      0.0394
Recall@0.5%:         0.0581
F1 @ threshold:       0.0469
Threshold (score):    -0.0326
Confusion matrix (overall, @ threshold):
[[1238485    6001]
 [   3991     246]]
Confusion matrix by type:
  CASH_OUT: [[421374, 2], [2112, 0]]
  TRANSFER: [[101840, 4474], [1879, 246]]
------------------------------------------------------------
Generalization check (training window vs held-out window)
ROC-AUC         train 0.8508  test 0.8538  gap -0.0030
PR-AUC          train 0.0063  test 0.0280  (base rates differ -- context only)
PR-AUC lift     train 8.18x  test 8.24x  (PR-AUC / base rate)
OK: no material train/test gap.
============================================================
```

| Metric | Training window (steps 1–355) | Held-out window (steps 356–743) |
|---|---|---|
| ROC-AUC | 0.8508 | **0.8538** (gap −0.003) |
| PR-AUC | 0.0063 | 0.0280 |
| Base fraud rate | 0.077% | 0.339% |
| PR-AUC lift (PR-AUC / base rate) | 8.18× | 8.24× |
| Precision@0.5% / Recall@0.5% | — | 0.0394 / 0.0581 |

**The model does not overfit.** Held-out ROC-AUC is as good as the
training-window ROC-AUC, and lift is almost identical. This is what
Isolation Forest's design predicts: each tree is built from only 256
sampled rows and the label is never seen, so there is little room to
memorize the training period.

**Its real weakness is `CASH_OUT`.** All 246 frauds caught in the held-out
window are `TRANSFER`s; **none of the 2,112 `CASH_OUT` frauds are flagged**
at the top-0.5% threshold. The current features don't separate fraudulent
`CASH_OUT`s from normal ones, so this is missing signal (underfitting),
not overfitting. Improving recall means adding features that describe
`CASH_OUT` fraud, not adding regularization.

Errors still concentrate in `TRANSFER` and `CASH_OUT`, the two types where
fraud lives in PaySim (see [02-dataset-and-eda.md](02-dataset-and-eda.md)),
rather than spreading across `CASH_IN`/`DEBIT`/`PAYMENT`, which would have
pointed to a mis-specified feature.

## Earlier result (run 1, in-sample only)

Run 1 was trained on all 6,362,604 rows and evaluated on those same rows,
before the split existed. It is kept for comparison, but its numbers
describe how well the model fits its own training data, not how it
performs on new transactions.

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

Against the full-dataset base rate of 0.129% (`8,197 / 6,362,604`), a
PR-AUC of 0.0114 is roughly an **8.8× enrichment** over random guessing.
Run 2's held-out ROC-AUC (0.8538) matches run 1's in-sample ROC-AUC
(0.8540), which confirms that run 1's numbers weren't inflated by
in-sample evaluation. Its precision/recall@k aren't directly comparable
with run 2's, because they are taken over a different set of rows with a
different fraud rate.

## Reading `model_runs` from SQL

```sql
SELECT run_id, trained_at, train_step_end,
       pr_auc, roc_auc, precision_at_k, recall_at_k, f1_at_threshold,
       train_pr_auc, train_roc_auc,
       train_roc_auc - roc_auc AS roc_auc_gap,
       is_deployed
FROM model_runs
ORDER BY run_id DESC;
```

`pr_auc`/`roc_auc` and the other headline columns are held-out metrics.
`train_pr_auc`/`train_roc_auc` are the same model scored on its training
window; both are `NULL` (as is `train_step_end`) for run 1, which predates
the split.

This is the query the dashboard's Model Performance view
(`sql/queries.py::get_model_run_history`) runs to build its PR-AUC/ROC-AUC
trend chart across model versions.

---
[← 06 — Modeling](06-modeling.md) | [Index](README.md) | Next: [08 — Drift Monitoring →](08-drift-monitoring.md)
