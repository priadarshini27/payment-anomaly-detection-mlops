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
