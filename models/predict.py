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
