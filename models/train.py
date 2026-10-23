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
