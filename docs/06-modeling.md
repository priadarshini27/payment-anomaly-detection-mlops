# 06 — Modeling

[← 05 — SQL Analytics](05-sql-analytics.md) | [Index](README.md) | Next: [07 — Evaluation Metrics →](07-evaluation-metrics.md)

## Why Isolation Forest

Isolation Forest was chosen over supervised alternatives (logistic
regression, gradient-boosted trees) because it does not require fraud
labels at training time, and because its underlying mechanism —
isolating points via random recursive partitioning — naturally assigns
short average path lengths to points that are "few and different," which
matches the profile of anomalous transactions in this dataset. Given the
extreme class imbalance documented in
[02-dataset-and-eda.md](02-dataset-and-eda.md) (0.13% fraud rate), a
supervised model trained on so few positive examples tends to overfit to
the specific fraud patterns present in the PaySim simulation rather than
generalizing to novel anomaly types — an unsupervised detector sidesteps
that risk entirely by never seeing the label.

## The feature set

The model is trained and scored on exactly 12 columns
(`MODEL_FEATURE_COLUMNS` in `preprocessing/feature_engineering.py`):

```
logAmount, balanceDeltaOrig, balanceDeltaDest, origZeroedOut,
isMerchantDest, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT,
type_TRANSFER, txnVelocity1h, txnVelocity24h
```

`isFraud` and `isFlaggedFraud` are never included — that's what makes
this an unsupervised detector rather than a supervised classifier wearing
one.

## Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Stabilizes the path-length average beyond sklearn's default of 100, reducing score variance across runs. |
| `contamination` | 0.0013 | Set to the empirical fraud rate observed in the training partition rather than sklearn's default `'auto'`, so the decision threshold reflects the real base rate. |
| `max_samples` | `'auto'` (256) | Subsampling per tree keeps isolation paths short and training tractable at multi-million-row scale. |
| `max_features` | 1.0 | All 12 engineered features are used per split; the feature set is small enough that sub-sampling features is unnecessary. |
| `random_state` | 42 | Fixed for reproducibility across training/evaluation/retraining runs. |

Features are standardized with `sklearn.preprocessing.StandardScaler`
before fitting, since Isolation Forest's random split thresholds are
sensitive to feature scale (an unscaled `amount`-derived feature spanning
orders of magnitude would dominate splits over a 0/1 one-hot column).

## Train/test split

The model is fit only on the **earliest 80% of rows by `step`**. Every
later row is held out: the scaler, the forest and the drift reference
window never see it, so [07-evaluation-metrics.md](07-evaluation-metrics.md)
can measure how the model performs on transactions from *after* its
training period, rather than on data it was fitted to.

| Partition | Steps | Rows | Fraud | Base fraud rate |
|---|---|---|---|---|
| Training | 1–355 | 5,113,881 | 3,960 | 0.077% |
| Held-out | 356–743 | 1,248,723 | 4,237 | 0.339% |

Design decisions:

- **Split by time, not at random.** A deployed model only ever scores
  transactions that happen after it was trained. A random split would mix
  later behaviour into training, and because `txnVelocity1h/24h` are built
  from each account's history, it would also put rows that share history
  on both sides of the split.
- **Split on whole steps.** `train_step_end_for_fraction()` picks the first
  step at which the cumulative row count reaches 80%, so each simulated
  hour lands entirely on one side. It always leaves at least one step held
  out.
- **Split by row count, not step count.** PaySim's volume collapses after
  roughly step 400 while fraud keeps arriving at a steady pace, so 80% of
  *rows* is reached at step 355 (less than half the 743 steps). The held-out
  window therefore has a much higher fraud rate. That doesn't hurt the
  evaluation, but it does mean raw PR-AUC can't be compared across the two
  windows (see [07-evaluation-metrics.md](07-evaluation-metrics.md)).
- **The cutoff is recorded.** `model_runs.train_step_end` stores the last
  training step for every run, so `models/evaluate.py` always knows which
  rows are held out for a given model. Databases created before this column
  existed get it added automatically by `ensure_split_columns()`.

The fraction can be changed with `--train-fraction` (default `0.8`).

## `models/train.py`

### File: models/train.py

```python
"""
models/train.py
==================
Fits an unsupervised Isolation Forest on the engineered transaction
features (never on isFraud), persists the fitted model + feature scaler
via joblib, captures the training feature distribution as the fixed
"reference window" used later by drift monitoring, and inserts a row into
model_runs recording the hyperparameters used.

Hyperparameters (Document 1, Table 2):
  n_estimators=200      -- stabilizes path-length averages beyond sklearn's
                            default of 100, reducing score variance.
  contamination=0.0013  -- set to the empirical fraud rate in the training
                            partition rather than sklearn's 'auto', so the
                            decision threshold reflects the real base rate.
  max_samples='auto'    -- 256 rows per tree; keeps isolation paths short
                            and training tractable at multi-million-row scale.
  max_features=1.0       -- all 12 engineered features used per split; the
                            feature set is small enough that sub-sampling
                            features is unnecessary.
  random_state=42        -- fixed for reproducibility across train/eval/retrain.

Train/test split:
  The model is fit only on the earliest TRAIN_FRACTION (80%) of rows by
  `step` (time order). Everything after `train_step_end` is held out and
  never touches the scaler, the model, or the drift reference window, so
  models/evaluate.py can measure out-of-time generalization rather than
  in-sample fit. A time split (not a random one) is used because a
  deployed model only ever scores transactions from after its training
  window, and a random split would leak future behaviour into training.

Run directly:
    python models/train.py
Reads the training partition of the `transactions` table from
database/database.db (which must already have txnVelocity1h/24h backfilled
by sql/queries.py), fits the model, and writes artifacts to
models/artifacts/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_engineering import MODEL_FEATURE_COLUMNS, build_model_matrix  # noqa: E402
from sql.queries import get_engine  # noqa: E402

ARTIFACTS_DIR = Path("models/artifacts")

TRAIN_FRACTION = 0.8  # earliest 80% of rows (by step) are used for training

HYPERPARAMS = dict(
    n_estimators=200,
    contamination=0.0013,
    max_samples="auto",
    max_features=1.0,
    random_state=42,
)


def train_step_end_for_fraction(step_counts: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> int:
    """
    Returns the last `step` of the training partition: the smallest step at
    which the cumulative row count reaches train_fraction of all rows.
    step_counts has columns [step, n] (one row per distinct step).
    Splitting on whole steps keeps every hour entirely on one side.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if len(step_counts) < 2:
        raise ValueError("Need at least two distinct steps to make a train/test split.")
    counts = step_counts.sort_values("step")
    cum_frac = counts["n"].cumsum() / counts["n"].sum()
    idx = int(np.searchsorted(cum_frac.to_numpy(), train_fraction))
    idx = min(idx, len(counts) - 2)  # always leave at least one step held out
    return int(counts["step"].iloc[idx])


def compute_train_step_end(engine, train_fraction: float = TRAIN_FRACTION) -> int:
    step_counts = pd.read_sql(
        text("SELECT step, COUNT(*) AS n FROM transactions GROUP BY step;"), engine
    )
    return train_step_end_for_fraction(step_counts, train_fraction)


def ensure_split_columns(engine) -> None:
    """
    Adds the train/test split columns to model_runs on databases created
    before they existed in database/schema.sql (CREATE TABLE IF NOT EXISTS
    won't alter an existing table).
    """
    new_cols = {"train_step_end": "INTEGER", "train_pr_auc": "REAL", "train_roc_auc": "REAL"}
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(model_runs);"))}
        for col, col_type in new_cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE model_runs ADD COLUMN {col} {col_type};"))


def load_training_data(engine, max_step: int | None = None, fetch_batch_size: int = 200_000) -> pd.DataFrame:
    """
    Loads every transaction row needed to build the model feature matrix.

    Every selected column is numeric, so this streams through the raw
    DB-API cursor in batches (fetchmany), converting each batch straight
    to a typed float64 numpy array and appending to a pre-allocated output
    array -- rather than pandas.read_sql(), which builds one large
    intermediate Python-object array holding every row as a tuple before
    typed conversion. At PaySim's full scale (6,362,604 rows x 15 columns)
    that intermediate object array is large enough (~730 MiB, on top of
    the final typed array) to raise numpy._core._exceptions.ArrayMemoryError
    on a memory-constrained machine. Streaming in fixed-size batches caps
    the peak extra memory at one batch's worth of Python tuples rather than
    the whole table's.

    max_step restricts the load to the training partition (step <= max_step).
    """
    cols = ["txn_id", "step", "isFraud"] + MODEL_FEATURE_COLUMNS
    where, params = "", ()
    if max_step is not None:
        where, params = " WHERE step <= ?", (int(max_step),)
    sql = f"SELECT {', '.join(cols)} FROM transactions{where};"

    raw_conn = engine.raw_connection()
    try:
        cur = raw_conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM transactions{where};", params)
        n_rows = cur.fetchone()[0]

        data = np.empty((n_rows, len(cols)), dtype="float64")
        cur.execute(sql, params)
        offset = 0
        while True:
            batch = cur.fetchmany(fetch_batch_size)
            if not batch:
                break
            data[offset:offset + len(batch)] = np.asarray(batch, dtype="float64")
            offset += len(batch)
    finally:
        raw_conn.close()

    return pd.DataFrame(data, columns=cols)


def fit_isolation_forest(X_scaled, hyperparams: dict = HYPERPARAMS) -> IsolationForest:
    model = IsolationForest(n_jobs=-1, **hyperparams)
    model.fit(X_scaled)
    return model


def save_artifacts(model, scaler, reference_window: pd.DataFrame, timestamp: str) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"model_{timestamp}.joblib"
    scaler_path = ARTIFACTS_DIR / f"scaler_{timestamp}.joblib"
    reference_path = ARTIFACTS_DIR / f"reference_window_{timestamp}.csv"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    reference_window.to_csv(reference_path, index=False)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "reference_window_path": str(reference_path),
    }


def insert_model_run(engine, timestamp: str, n_rows: int, artifact_paths: dict,
                     train_step_end: int | None = None) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO model_runs (
                    trained_at, n_estimators, contamination, max_samples,
                    max_features, random_state, n_training_rows,
                    model_path, scaler_path, reference_window_path,
                    train_step_end, is_deployed
                ) VALUES (
                    :trained_at, :n_estimators, :contamination, :max_samples,
                    :max_features, :random_state, :n_training_rows,
                    :model_path, :scaler_path, :reference_window_path,
                    :train_step_end, 0
                );
                """
            ),
            {
                "trained_at": timestamp,
                "n_estimators": HYPERPARAMS["n_estimators"],
                "contamination": HYPERPARAMS["contamination"],
                "max_samples": str(HYPERPARAMS["max_samples"]),
                "max_features": HYPERPARAMS["max_features"],
                "random_state": HYPERPARAMS["random_state"],
                "n_training_rows": n_rows,
                "train_step_end": train_step_end,
                **artifact_paths,
            },
        )
        run_id = result.lastrowid
    return run_id


def train(db_path: str | None = None, reference_sample_size: int = 200_000,
          train_fraction: float = TRAIN_FRACTION) -> int:
    """
    Runs the full training routine end-to-end and returns the new run_id.

    reference_sample_size caps how many training rows are persisted as the
    drift-monitoring reference window (the full training set is used to
    *fit* the model, but only a representative sample needs to be kept on
    disk as the reference distribution for PSI/KS comparisons later).

    train_fraction sets the time-based split: only the earliest
    train_fraction of rows (by step) are loaded and fitted on; the rest are
    held out for models/evaluate.py.
    """
    from sql.queries import get_engine as _get_engine, DEFAULT_DB_PATH

    engine = _get_engine(db_path) if db_path else _get_engine(DEFAULT_DB_PATH)

    ensure_split_columns(engine)
    with engine.connect() as conn:
        if conn.execute(text("SELECT COUNT(*) FROM transactions;")).scalar() == 0:
            raise RuntimeError(
                "No rows found in `transactions`. Run preprocessing/clean_data.py and "
                "database/load_data.py first."
            )
    train_step_end = compute_train_step_end(engine, train_fraction)

    print(f"Loading training data from SQLite (step <= {train_step_end})...")
    df = load_training_data(engine, max_step=train_step_end)
    print(f"Loaded {len(df):,} training rows; steps > {train_step_end} are held out.")

    X = build_model_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Fitting IsolationForest "
          f"(n_estimators={HYPERPARAMS['n_estimators']}, "
          f"contamination={HYPERPARAMS['contamination']})...")
    model = fit_isolation_forest(X_scaled)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    reference_window = df.sample(
        n=min(reference_sample_size, len(df)), random_state=HYPERPARAMS["random_state"]
    )[["step"] + MODEL_FEATURE_COLUMNS]

    artifact_paths = save_artifacts(model, scaler, reference_window, timestamp)
    run_id = insert_model_run(engine, timestamp, len(df), artifact_paths, train_step_end)

    print(f"Training complete. model_runs.run_id = {run_id}")
    print(f"  model:            {artifact_paths['model_path']}")
    print(f"  scaler:           {artifact_paths['scaler_path']}")
    print(f"  reference window: {artifact_paths['reference_window_path']}")

    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--reference-sample-size", type=int, default=200_000)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION,
                        help="Fraction of rows (earliest by step) used for training; the rest is held out.")
    args = parser.parse_args()
    train(db_path=args.db_path, reference_sample_size=args.reference_sample_size,
          train_fraction=args.train_fraction)


if __name__ == "__main__":
    main()
```

## `models/predict.py`

### File: models/predict.py

```python
"""
models/predict.py
====================
Loads a persisted Isolation Forest model (+ its scaler) and scores
transactions, producing a continuous anomaly_score and a binary is_anomaly
flag at a chosen threshold.

Used two ways:
  * As a library from evaluate.py / drift/retrain.py, to score a DataFrame
    already in memory.
  * As a CLI (`python models/predict.py`) that scores every row currently
    in the `transactions` table and writes anomaly_score/is_anomaly back
    into SQLite.

sklearn's IsolationForest.decision_function returns *higher* values for
*more normal* points and lower/negative values for anomalies. This module
negates that so anomaly_score follows the more intuitive convention: higher
= more anomalous.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_engineering import build_model_matrix  # noqa: E402
from sql.queries import get_engine, DEFAULT_DB_PATH  # noqa: E402


def load_model_and_scaler(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def score(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Scores a DataFrame that already has engineered features (and, ideally,
    velocity features) present. Returns the input df with two new columns:
    anomaly_score (higher = more anomalous) and is_anomaly (model's own
    contamination-based prediction, 1 = anomaly / -1->1 remapped).
    """
    df = df.copy()
    X = build_model_matrix(df)
    X_scaled = scaler.transform(X)

    # decision_function: higher = more normal. Negate so higher = more anomalous.
    df["anomaly_score"] = -model.decision_function(X_scaled)
    # predict(): 1 = inlier, -1 = outlier (per sklearn convention). Remap to 0/1.
    raw_pred = model.predict(X_scaled)
    df["is_anomaly"] = (raw_pred == -1).astype(int)
    return df


def score_and_persist(db_path: str | None = None, model_path: str | None = None,
                       scaler_path: str | None = None, run_id: int | None = None,
                       read_chunksize: int = 200_000, write_batch_size: int = 200_000) -> int:
    """
    Scores every row in `transactions` using the specified model (or the
    latest trained model_runs row if not specified), and writes
    anomaly_score/is_anomaly back into the table. Returns the number of
    rows scored.

    At PaySim's full scale (6,362,604 rows), two naive approaches don't
    scale (see sql/queries.py::get_velocity_features for the same lesson
    learned there):
      * `pd.read_sql("SELECT * FROM transactions")` builds one large
        intermediate Python-object array before typed conversion, which
        can exhaust memory on top of the final DataFrame + model scoring
        arrays. Reading in chunks via `pd.read_sql(..., chunksize=...)`
        caps peak memory to one chunk at a time.
      * Writing scores back with one `UPDATE ... WHERE txn_id = ?` per row
        is dominated by per-statement overhead at millions of rows.
        Instead, results are loaded into a temp table via a single bulk
        `executemany` INSERT, then applied with one set-based
        correlated-subquery UPDATE.
    """
    engine = get_engine(db_path or DEFAULT_DB_PATH)

    if model_path is None or scaler_path is None:
        with engine.connect() as conn:
            if run_id is not None:
                row = conn.execute(
                    text("SELECT model_path, scaler_path FROM model_runs WHERE run_id = :rid"),
                    {"rid": run_id},
                ).fetchone()
            else:
                row = conn.execute(
                    text("SELECT model_path, scaler_path FROM model_runs ORDER BY run_id DESC LIMIT 1")
                ).fetchone()
        if row is None:
            raise RuntimeError("No trained model found in model_runs. Run models/train.py first.")
        model_path, scaler_path = row

    model, scaler = load_model_and_scaler(model_path, scaler_path)

    scored_chunks = []
    for chunk in pd.read_sql(text("SELECT * FROM transactions;"), engine, chunksize=read_chunksize):
        scored_chunks.append(score(chunk, model, scaler)[["txn_id", "anomaly_score", "is_anomaly"]])
    scored = pd.concat(scored_chunks, ignore_index=True)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _score_tmp;"))
        conn.execute(
            text(
                "CREATE TEMP TABLE _score_tmp "
                "(txn_id INTEGER PRIMARY KEY, anomaly_score REAL, is_anomaly INTEGER);"
            )
        )
        raw_conn = conn.connection
        cur = raw_conn.cursor()
        params = list(
            zip(
                scored["txn_id"].astype(int),
                scored["anomaly_score"].astype(float),
                scored["is_anomaly"].astype(int),
            )
        )
        for i in range(0, len(params), write_batch_size):
            cur.executemany(
                "INSERT INTO _score_tmp (txn_id, anomaly_score, is_anomaly) VALUES (?, ?, ?);",
                params[i:i + write_batch_size],
            )

        conn.execute(
            text(
                """
                UPDATE transactions
                SET anomaly_score = (SELECT anomaly_score FROM _score_tmp WHERE _score_tmp.txn_id = transactions.txn_id),
                    is_anomaly    = (SELECT is_anomaly    FROM _score_tmp WHERE _score_tmp.txn_id = transactions.txn_id)
                WHERE txn_id IN (SELECT txn_id FROM _score_tmp);
                """
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS _score_tmp;"))

    print(f"Scored and persisted anomaly_score/is_anomaly for {len(scored):,} rows.")
    return len(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--scaler-path", default=None)
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args()

    score_and_persist(
        db_path=args.db_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
```

## Running training

```bash
python models/train.py                      # default: train on earliest 80% of rows
python models/train.py --train-fraction 0.7 # hold out more data
```

This must run **after** `sql/queries.py` has backfilled the velocity
features — otherwise `txnVelocity1h`/`txnVelocity24h` would be `NULL` for
every row, and `build_model_matrix()` would silently default them to 0
(see the fallback logic in `preprocessing/feature_engineering.py`),
degrading two of the twelve model features to noise.

Training on the 5.1M-row training partition completes quickly because
`max_samples='auto'` caps each of the 200 trees at 256 subsampled rows —
Isolation Forest's whole design point is that it does not need to see
every row during a single tree's construction to isolate anomalies
effectively. The dominant cost is loading and typing the feature matrix,
not fitting the forest itself — see the note on `load_training_data`
above.

### Measured result (run 2, time-based split)

```
Loading training data from SQLite (step <= 355)...
Loaded 5,113,881 training rows; steps > 355 are held out.
Fitting IsolationForest (n_estimators=200, contamination=0.0013)...
Training complete. model_runs.run_id = 2
  model:            models\artifacts\model_20261017T025742Z.joblib
  scaler:           models\artifacts\scaler_20261017T025742Z.joblib
  reference window: models\artifacts\reference_window_20261017T025742Z.csv
```

### Earlier result (run 1, before the split existed)

Run 1 was fit on every row, so it has no held-out data. It is kept in
`model_runs` for comparison; `models/evaluate.py` still evaluates it, but
prints a warning that its metrics are in-sample only.

```
Loading training data from SQLite...
Loaded 6,362,604 rows.
Fitting IsolationForest (n_estimators=200, contamination=0.0013)...
Training complete. model_runs.run_id = 1
  model:            models\artifacts\model_20260802T131825Z.joblib
  scaler:           models\artifacts\scaler_20260802T131825Z.joblib
  reference window: models\artifacts\reference_window_20260802T131825Z.csv

real    2m57.760s
```

### A memory lesson learned the hard way

The first version of `load_training_data` used `pandas.read_sql(sql,
engine)` directly. Against the full dataset this raised
`numpy._core._exceptions.ArrayMemoryError: Unable to allocate 728 MiB for
an array with shape (6362604, 15) and data type object` — `read_sql`
builds an intermediate array of Python object tuples (one per row) before
converting to typed columns, and at this scale that intermediate
representation alone needs hundreds of megabytes on top of the final
array. The fix, shown above, streams through the raw DB-API cursor in
batches and writes directly into a pre-allocated typed `float64` array,
never materializing the full object-tuple intermediate. The same fix was
applied to `models/predict.py` and `models/evaluate.py` (see
[07-evaluation-metrics.md](07-evaluation-metrics.md)), which have the
identical `SELECT * FROM transactions` pattern.

## Model versioning

Every call to `train()` inserts a new `model_runs` row and writes three
new artifacts to `models/artifacts/`, timestamped to the second
(`model_<UTC timestamp>.joblib`, `scaler_<...>.joblib`,
`reference_window_<...>.csv`), along with the `train_step_end` cutoff
that defines its held-out window. Nothing is overwritten — every trained
version is kept, which is what lets
[06-modeling.md](06-modeling.md)/[07-evaluation-metrics.md](07-evaluation-metrics.md)'s
model-comparison story and the dashboard's PR-AUC trend chart work
directly off SQL rather than ad-hoc notebook output. `is_deployed` is set
to 1 on exactly one `model_runs` row at a time by `models/evaluate.py`
(see next document) — the most recently evaluated model becomes the
"live" one the dashboard and `drift/retrain.py` treat as current.

---
[← 05 — SQL Analytics](05-sql-analytics.md) | [Index](README.md) | Next: [07 — Evaluation Metrics →](07-evaluation-metrics.md)
