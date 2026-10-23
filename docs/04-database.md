# 04 — Database

[← 03 — Data Cleaning](03-data-cleaning.md) | [Index](README.md) | Next: [05 — SQL Analytics →](05-sql-analytics.md)

The project uses **SQLite** as its single persistence layer — simple to
set up, zero-configuration, and entirely sufficient for a dataset in the
single-digit-millions-of-rows range. Four tables back the full lifecycle:

| Table | Written by | Purpose |
|---|---|---|
| `transactions` | `database/load_data.py`, backfilled by `sql/queries.py` and `models/predict.py` | Cleaned, feature-engineered PaySim transaction log |
| `model_runs` | `models/train.py` (hyperparameters, `train_step_end` split cutoff), `models/evaluate.py` (held-out and training-window metrics, UPDATE) | One row per trained Isolation Forest version |
| `drift_metrics` | `drift/retrain.py` | Per-feature PSI/KS statistics for every drift check |
| `retrain_log` | `drift/retrain.py` | Audit trail of every retraining decision, triggered or not |

## `database/schema.sql`

### File: database/schema.sql

```sql
-- =============================================================================
-- Payment Anomaly Detection MLOps -- SQLite Schema
-- =============================================================================
-- This schema backs the full lifecycle of the project:
--   * `transactions`    -- cleaned, feature-engineered PaySim transaction log
--   * `model_runs`      -- one row per trained Isolation Forest model version
--   * `drift_metrics`   -- per-feature PSI / KS drift statistics per drift check
--   * `retrain_log`     -- audit trail of automated retraining decisions
--
-- Run via: sqlite3 database/database.db < database/schema.sql
-- or programmatically through database/load_data.py (which executes this
-- file automatically on first run).
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- transactions
-- -----------------------------------------------------------------------------
-- One row per cleaned PaySim transaction. Raw PaySim columns are preserved
-- as-is; engineered columns (balanceDeltaOrig .. type_TRANSFER) are computed
-- by preprocessing/feature_engineering.py before load. txnVelocity1h/24h are
-- NULL immediately after load and are backfilled by sql/queries.py's
-- get_velocity_features(), since they require a window function over the
-- full ordered history rather than a single-row computation.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    txn_id            INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Raw PaySim fields
    step              INTEGER NOT NULL,
    type              TEXT    NOT NULL,
    amount            REAL    NOT NULL,
    nameOrig          TEXT    NOT NULL,
    oldbalanceOrg     REAL    NOT NULL,
    newbalanceOrig    REAL    NOT NULL,
    nameDest          TEXT    NOT NULL,
    oldbalanceDest    REAL    NOT NULL,
    newbalanceDest    REAL    NOT NULL,
    isFraud           INTEGER NOT NULL,
    isFlaggedFraud    INTEGER NOT NULL,

    -- Engineered features (preprocessing/feature_engineering.py)
    balanceDeltaOrig  REAL,
    balanceDeltaDest  REAL,
    origZeroedOut     INTEGER,
    logAmount         REAL,
    isMerchantDest    INTEGER,
    type_CASH_IN      INTEGER,
    type_CASH_OUT     INTEGER,
    type_DEBIT        INTEGER,
    type_PAYMENT      INTEGER,
    type_TRANSFER     INTEGER,

    -- Backfilled post-load by sql/queries.py::get_velocity_features()
    txnVelocity1h     INTEGER,
    txnVelocity24h    INTEGER,

    -- Backfilled post-scoring by models/predict.py
    anomaly_score     REAL,
    is_anomaly        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_transactions_step     ON transactions (step);
CREATE INDEX IF NOT EXISTS idx_transactions_nameOrig ON transactions (nameOrig);
CREATE INDEX IF NOT EXISTS idx_transactions_type     ON transactions (type);
CREATE INDEX IF NOT EXISTS idx_transactions_isFraud  ON transactions (isFraud);

-- -----------------------------------------------------------------------------
-- model_runs
-- -----------------------------------------------------------------------------
-- One row per trained model version. Hyperparameters are written by
-- models/train.py at training time; the evaluation metric columns are filled
-- in afterward by models/evaluate.py (UPDATE by run_id). model_path /
-- scaler_path / reference_window_path point at the joblib/parquet artifacts
-- persisted under models/artifacts/.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at              TEXT    NOT NULL,          -- ISO-8601 timestamp

    -- Hyperparameters (Doc1 Table 2)
    n_estimators            INTEGER NOT NULL,
    contamination           REAL    NOT NULL,
    max_samples             TEXT    NOT NULL,
    max_features            REAL    NOT NULL,
    random_state            INTEGER NOT NULL,
    n_training_rows         INTEGER,

    -- Artifact locations
    model_path              TEXT    NOT NULL,
    scaler_path             TEXT    NOT NULL,
    reference_window_path   TEXT,

    -- Time-based train/test split: rows with step <= train_step_end were
    -- used for fitting; later rows are the held-out evaluation window.
    train_step_end          INTEGER,

    -- Evaluation metrics (filled in by models/evaluate.py). These are
    -- computed on the held-out window (step > train_step_end).
    pr_auc                  REAL,
    roc_auc                 REAL,
    precision_at_k          REAL,
    recall_at_k             REAL,
    f1_at_threshold         REAL,
    threshold                REAL,
    k_fraction               REAL,

    -- Same ranking metrics on the training window, for the train/test gap.
    train_pr_auc             REAL,
    train_roc_auc            REAL,

    is_deployed              INTEGER NOT NULL DEFAULT 0,
    notes                    TEXT
);

-- -----------------------------------------------------------------------------
-- drift_metrics
-- -----------------------------------------------------------------------------
-- One row per (drift-check run, feature). Written by drift/retrain.py after
-- computing PSI (drift/psi.py) and the KS test (drift/ks_test.py) for every
-- monitored feature, comparing a fixed reference window (captured at
-- training time) against a current scoring window.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   INTEGER NOT NULL REFERENCES model_runs (run_id),
    feature_name             TEXT    NOT NULL,

    psi_value                REAL    NOT NULL,
    ks_statistic             REAL    NOT NULL,
    ks_pvalue                REAL    NOT NULL,

    reference_window_start   INTEGER NOT NULL,   -- step
    reference_window_end     INTEGER NOT NULL,   -- step
    current_window_start     INTEGER NOT NULL,   -- step
    current_window_end       INTEGER NOT NULL,   -- step

    computed_at              TEXT    NOT NULL,   -- ISO-8601 timestamp
    psi_flag                 TEXT    NOT NULL,   -- 'none' | 'moderate' | 'significant'
    ks_flag                  INTEGER NOT NULL    -- 1 if ks_pvalue < 0.05 else 0
);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_run_id  ON drift_metrics (run_id);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature ON drift_metrics (feature_name);

-- -----------------------------------------------------------------------------
-- retrain_log
-- -----------------------------------------------------------------------------
-- Audit trail of every drift check's retraining decision, whether or not it
-- fired. Written by drift/retrain.py. resulting_run_id is populated only
-- when a retrain was actually triggered and completed.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrain_log (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at            TEXT    NOT NULL,   -- ISO-8601 timestamp
    triggered             INTEGER NOT NULL,   -- 0/1
    trigger_reason        TEXT,               -- 'psi' | 'performance' | 'psi+performance' | NULL
    n_features_drifted    INTEGER NOT NULL,
    pr_auc_drop_pct       REAL,
    baseline_run_id       INTEGER REFERENCES model_runs (run_id),
    resulting_run_id      INTEGER REFERENCES model_runs (run_id)
);
```

## `database/load_data.py`

### File: database/load_data.py

```python
"""
database/load_data.py
========================
Creates the SQLite database (if needed) from database/schema.sql, and loads
the cleaned, feature-engineered PaySim dataset (data/paysim_clean.csv,
produced by preprocessing/clean_data.py) into the `transactions` table in
chunked batches via executemany.

Run directly:
    python database/load_data.py
Requires data/paysim_clean.csv to already exist (run
preprocessing/clean_data.py first, or just run clean_data.py which calls
this module automatically).
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path("database/database.db")
DEFAULT_SCHEMA_PATH = Path("database/schema.sql")
DEFAULT_CLEAN_CSV_PATH = Path("data/paysim_clean.csv")
DEFAULT_CHUNKSIZE = 100_000

# Columns present in paysim_clean.csv, in the order they are inserted into
# the `transactions` table (must match the non-autoincrement, non-backfilled
# columns of database/schema.sql).
INSERT_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
    "balanceDeltaOrig", "balanceDeltaDest", "origZeroedOut", "logAmount",
    "isMerchantDest", "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER",
]

INSERT_SQL = (
    "INSERT INTO transactions (" + ", ".join(INSERT_COLUMNS) + ") VALUES ("
    + ", ".join("?" for _ in INSERT_COLUMNS) + ")"
)


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_schema(conn: sqlite3.Connection, schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> None:
    """Executes schema.sql against the given connection (idempotent -- uses IF NOT EXISTS)."""
    schema_sql = Path(schema_path).read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()


def reset_transactions_table(conn: sqlite3.Connection) -> None:
    """Clears any existing transaction rows so re-running load_data.py is idempotent."""
    conn.execute("DELETE FROM transactions;")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'transactions';")
    conn.commit()


def load_clean_csv_to_sqlite(
    clean_csv_path: str | Path = DEFAULT_CLEAN_CSV_PATH,
    db_path: str | Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    reset: bool = True,
) -> int:
    """
    Loads data/paysim_clean.csv into the `transactions` table in chunked
    batches. Returns the total number of rows inserted.
    """
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    clean_csv_path = Path(clean_csv_path)
    if not clean_csv_path.exists():
        raise FileNotFoundError(
            f"{clean_csv_path} not found. Run preprocessing/clean_data.py first."
        )

    conn = get_connection(db_path)
    init_schema(conn)
    if reset:
        reset_transactions_table(conn)

    t0 = time.time()
    total_rows = 0
    cur = conn.cursor()
    for chunk in pd.read_csv(clean_csv_path, chunksize=chunksize):
        chunk = chunk[INSERT_COLUMNS]
        rows = list(chunk.itertuples(index=False, name=None))
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        total_rows += len(rows)
        print(f"  inserted {total_rows:,} rows so far...")

    elapsed = time.time() - t0
    print(f"Loaded {total_rows:,} rows into {db_path} in {elapsed:.1f}s.")
    conn.close()
    return total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", default=str(DEFAULT_CLEAN_CSV_PATH))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Append to the existing transactions table instead of clearing it first.",
    )
    args = parser.parse_args()

    load_clean_csv_to_sqlite(
        clean_csv_path=args.clean_csv,
        db_path=args.db_path,
        chunksize=args.chunksize,
        reset=not args.no_reset,
    )


if __name__ == "__main__":
    main()
```

## Running it

```bash
python database/load_data.py
```

This is idempotent by default (`reset=True`): re-running it clears the
`transactions` table first, so you never end up with duplicated rows from
re-running the loader. Pass `--no-reset` to append instead.

## Inspecting the database

```bash
sqlite3 database/database.db
```

```sql
.tables
.schema transactions
SELECT COUNT(*) FROM transactions;
SELECT type, COUNT(*), SUM(isFraud) FROM transactions GROUP BY type;
```

## Resetting the database

To start completely fresh (e.g. after changing the schema):

```bash
rm database/database.db          # macOS/Linux/Git Bash
Remove-Item database\database.db # PowerShell
python database/load_data.py     # recreates schema.sql tables and reloads
```

---
[← 03 — Data Cleaning](03-data-cleaning.md) | [Index](README.md) | Next: [05 — SQL Analytics →](05-sql-analytics.md)
