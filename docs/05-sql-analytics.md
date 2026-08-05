# 05 — SQL Analytics

[← 04 — Database](04-database.md) | [Index](README.md) | Next: [06 — Modeling →](06-modeling.md)

`sql/queries.py` is the single place parameterized SQL lives in this
project. It wraps a SQLAlchemy engine so the same functions are reusable
from a script, a notebook, or the Streamlit dashboard without duplicating
connection logic.

## Why velocity features are computed here, not in `feature_engineering.py`

`txnVelocity1h` and `txnVelocity24h` (count of a customer's transactions
in the preceding 1 / 24 hours) require visibility into the **full ordered
history per `nameOrig`** — something a chunked, 500k-row-at-a-time
cleaning pass cannot see, since a customer's transactions can span
multiple chunks. They are therefore computed once, after the full dataset
is loaded into `transactions`, via `get_velocity_features()`.

## Performance: two rejected implementations before this one

At PaySim's full scale (6,362,604 rows), the naive ways to compute and
write back a per-row windowed count are both too slow to be practical,
and the reasons why are worth documenting since they generalize to any
per-row backfill at multi-million-row scale:

1. **Two correlated SQL subqueries per row.** The first implementation
   computed both windows as SQL correlated subqueries
   (`SELECT COUNT(*) FROM transactions b WHERE b.nameOrig = a.nameOrig AND
   b.step BETWEEN ...`). Even though `nameOrig` is (empirically, on this
   dataset) almost 1:1 with row count — 6,353,291 distinct originators
   across 6,362,604 rows, at most 3 transactions per originator — SQLite
   still pays an index B-tree traversal *per row per subquery*: roughly
   12.7 million lookups. This did not finish in over 20 minutes of active
   CPU time and was killed.
2. **Fast vectorized computation, slow row-by-row write-back.** The
   second implementation fixed the computation — pull `(txn_id, nameOrig,
   step)` into pandas once, sort, and compute the windowed count per
   originator group with a vectorized `numpy.searchsorted` (O(n log n)
   total, no per-row SQL round-trip). The computation itself is fast. But
   writing the result back with a Python-level `executemany` of one
   `UPDATE ... WHERE txn_id = ?` per row is *itself* the bottleneck,
   independent of how the values were computed: 6.36 million individual
   parameterized statements, each with its own `WHERE`-clause row lookup,
   still did not finish in a reasonable time.

The lesson that shaped the final implementation: **at this scale, the
cost center is almost always the write pattern, not the read or the
computation.** A single set-based write beats millions of row-at-a-time
writes even when the per-row work is cheap.

## The final implementation

`get_velocity_features()` keeps the fast vectorized computation from
attempt 2 (`compute_velocity_dataframe`), but changes how the result is
written back:

1. Load the computed `(txn_id, v1h, v24h)` triples into a **temp table**
   via a single `executemany` **INSERT** — cheap, because an `INSERT`
   has no `WHERE`-clause row lookup, unlike an `UPDATE`.
2. Apply all 6.36 million values in **one set-based `UPDATE ... FROM`-style
   join** (a correlated-subquery `UPDATE` against the temp table, which
   SQLite executes as a single statement rather than millions of
   round-trips) instead of one `UPDATE` per row.

Measured on the full dataset: **4m25s total**, versus 20+ minutes with no
completion for approach 1, and no completion in a comparable window for
approach 2's write-back. This is the version that ships.

## `sql/queries.py`

### File: sql/queries.py

```python
"""
sql/queries.py
=================
Parameterized SQL analytics used by the modeling, drift, and dashboard
layers. Everything here talks to the SQLite database through a SQLAlchemy
engine so the same functions can be reused from a script, a notebook, or
the Streamlit dashboard without duplicating connection logic.

Functions:
  get_engine                       -- creates a SQLAlchemy engine for database.db
  compute_velocity_dataframe        -- vectorized numpy computation of the velocity windows
  get_velocity_features              -- computes and bulk-backfills txnVelocity1h/24h
  get_type_aggregates                 -- volume/value by transaction type
  get_fraud_breakdown                  -- fraud rate by type / amount band / step
  get_cohort_balance_consistency        -- balance-residual distribution query
  get_transaction_overview_timeseries    -- per-step transaction counts/volumes
  get_model_run_history                   -- model_runs history for trend charts
  get_drift_summary                        -- drift_metrics for a given run
  get_latest_deployed_run                   -- most recent model_runs row
  get_system_health                          -- row counts + last-run timestamps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_PATH = Path("database/database.db")

# Window sizes for velocity features, expressed in PaySim `step` units
# (1 step == 1 simulated hour, per Document 1 Section 2).
VELOCITY_WINDOW_1H_STEPS = 1
VELOCITY_WINDOW_24H_STEPS = 24


def get_engine(db_path: str | Path = DEFAULT_DB_PATH) -> Engine:
    """Creates a SQLAlchemy engine for the project's SQLite database."""
    return create_engine(f"sqlite:///{Path(db_path)}")


def _windowed_counts_for_group(steps: np.ndarray, window: int) -> np.ndarray:
    """
    For a single originator's sorted step array, returns, for each row, the
    count of that originator's transactions with step in
    [step - window, step] inclusive. Vectorized with searchsorted rather
    than a per-row scan: O(n log n) per group instead of O(n^2).
    """
    lower = np.searchsorted(steps, steps - window, side="left")
    upper = np.searchsorted(steps, steps, side="right")
    return upper - lower


def compute_velocity_dataframe(name_orig: np.ndarray, step: np.ndarray, txn_id: np.ndarray) -> pd.DataFrame:
    """
    Vectorized computation of txnVelocity1h/24h for arbitrary
    (txn_id, nameOrig, step) arrays. Sorts by (nameOrig, step) once, then
    computes the windowed count per originator group with numpy
    searchsorted -- O(n log n) total, no per-row SQL round-trip and no
    per-row Python-level loop over the full dataset. Used both by
    get_velocity_features() (post-hoc backfill against an existing table)
    and by preprocessing/clean_data.py (computed once up front, before the
    very first INSERT, so the column is never NULL/backfilled in the
    common case -- see that module for details).
    """
    order = np.lexsort((step, name_orig))
    name_sorted = name_orig[order]
    step_sorted = step[order]

    v1h_sorted = np.empty(len(order), dtype="int32")
    v24h_sorted = np.empty(len(order), dtype="int32")

    group_starts = np.flatnonzero(
        np.concatenate(([True], name_sorted[1:] != name_sorted[:-1]))
    )
    group_ends = np.concatenate((group_starts[1:], [len(order)]))

    for start, end in zip(group_starts, group_ends):
        group_steps = step_sorted[start:end]
        v1h_sorted[start:end] = _windowed_counts_for_group(group_steps, VELOCITY_WINDOW_1H_STEPS)
        v24h_sorted[start:end] = _windowed_counts_for_group(group_steps, VELOCITY_WINDOW_24H_STEPS)

    # Scatter back to original row order via the sort's inverse permutation.
    v1h = np.empty_like(v1h_sorted)
    v24h = np.empty_like(v24h_sorted)
    v1h[order] = v1h_sorted
    v24h[order] = v24h_sorted

    return pd.DataFrame({"txn_id": txn_id, "txnVelocity1h": v1h, "txnVelocity24h": v24h})


def get_velocity_features(engine: Engine) -> int:
    """
    Computes txnVelocity1h and txnVelocity24h per nameOrig and writes the
    results back into the `transactions` table.

    txnVelocity1h(row)  = count of that originator's transactions with
                           step in [row.step - 1, row.step] (inclusive),
                           i.e. the current + immediately preceding hour.
    txnVelocity24h(row) = same idea over the preceding 24-hour window.

    Implementation note: two earlier versions of this function were tried
    and rejected as too slow at PaySim's 6.36M-row scale:
      1. Two correlated SQL subqueries per row (SQLite pays an index-lookup
         cost per row per subquery -- did not finish in 20+ minutes).
      2. The vectorized numpy computation below, but written back with a
         Python-level executemany of one UPDATE per row -- computing the
         values was fast, but 6.36M individual parameterized UPDATE
         statements is itself the bottleneck regardless of how the values
         were computed (also did not finish in a reasonable time).

    This version keeps the fast vectorized computation (see
    compute_velocity_dataframe) but writes the result back as a single
    bulk operation: the computed values are loaded into a temp table via
    one fast executemany INSERT (cheap -- no WHERE-clause row lookup per
    statement), and then a single correlated-subquery UPDATE against that
    temp table applies all 6.36M values in one set-based statement instead
    of millions of round-trips. Measured on the full dataset: 4m25s total.

    For a fresh load, prefer computing velocity in
    preprocessing/clean_data.py before the initial INSERT instead of
    calling this function as a post-hoc backfill -- see that module.

    Returns the number of rows updated.
    """
    df = pd.read_sql(text("SELECT txn_id, nameOrig, step FROM transactions;"), engine)
    velocity_df = compute_velocity_dataframe(
        df["nameOrig"].to_numpy(), df["step"].to_numpy(), df["txn_id"].to_numpy()
    )

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _velocity_tmp;"))
        conn.execute(
            text(
                "CREATE TEMP TABLE _velocity_tmp "
                "(txn_id INTEGER PRIMARY KEY, v1h INTEGER, v24h INTEGER);"
            )
        )
        raw_conn = conn.connection
        cur = raw_conn.cursor()
        params = list(
            zip(
                velocity_df["txn_id"].astype(int),
                velocity_df["txnVelocity1h"].astype(int),
                velocity_df["txnVelocity24h"].astype(int),
            )
        )
        cur.executemany("INSERT INTO _velocity_tmp (txn_id, v1h, v24h) VALUES (?, ?, ?);", params)

        conn.execute(
            text(
                """
                UPDATE transactions
                SET txnVelocity1h  = (SELECT v1h  FROM _velocity_tmp WHERE _velocity_tmp.txn_id = transactions.txn_id),
                    txnVelocity24h = (SELECT v24h FROM _velocity_tmp WHERE _velocity_tmp.txn_id = transactions.txn_id)
                WHERE txn_id IN (SELECT txn_id FROM _velocity_tmp);
                """
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS _velocity_tmp;"))

    return len(velocity_df)


def get_type_aggregates(engine: Engine) -> pd.DataFrame:
    """Volume (count) and value (sum/avg amount) of transactions by type."""
    sql = text(
        """
        SELECT type,
               COUNT(*)         AS txn_count,
               SUM(amount)      AS total_amount,
               AVG(amount)      AS avg_amount,
               SUM(isFraud)     AS fraud_count
        FROM transactions
        GROUP BY type
        ORDER BY txn_count DESC;
        """
    )
    return pd.read_sql(sql, engine)


def get_fraud_breakdown(engine: Engine) -> dict[str, pd.DataFrame]:
    """
    Fraud rate broken down three ways, per Document 2 Section 6.8's
    guidance to never report a single aggregate fraud rate:
      by_type       -- fraud rate per transaction type
      by_amount_band -- fraud rate per log-scale amount bucket
      by_step_bucket  -- fraud rate per 24-step (1-day) bucket, for trend charts
    """
    by_type = pd.read_sql(
        text(
            """
            SELECT type,
                   COUNT(*)                          AS total_count,
                   SUM(isFraud)                       AS fraud_count,
                   1.0 * SUM(isFraud) / COUNT(*)      AS fraud_rate
            FROM transactions
            GROUP BY type
            ORDER BY fraud_rate DESC;
            """
        ),
        engine,
    )

    by_amount_band = pd.read_sql(
        text(
            """
            SELECT
                CASE
                    WHEN amount < 1000        THEN '< 1K'
                    WHEN amount < 10000       THEN '1K-10K'
                    WHEN amount < 100000      THEN '10K-100K'
                    WHEN amount < 1000000     THEN '100K-1M'
                    ELSE '>= 1M'
                END AS amount_band,
                COUNT(*)                     AS total_count,
                SUM(isFraud)                  AS fraud_count,
                1.0 * SUM(isFraud) / COUNT(*)  AS fraud_rate
            FROM transactions
            GROUP BY amount_band;
            """
        ),
        engine,
    )

    by_step_bucket = pd.read_sql(
        text(
            """
            SELECT (step / 24)              AS day,
                   COUNT(*)                  AS total_count,
                   SUM(isFraud)               AS fraud_count,
                   1.0 * SUM(isFraud) / COUNT(*) AS fraud_rate
            FROM transactions
            GROUP BY day
            ORDER BY day;
            """
        ),
        engine,
    )

    return {"by_type": by_type, "by_amount_band": by_amount_band, "by_step_bucket": by_step_bucket}


def get_cohort_balance_consistency(engine: Engine) -> pd.DataFrame:
    """
    Balance-residual distribution (balanceDeltaOrig), split by isFraud, to
    validate the Document 1 Section 4.1 claim that non-zero residuals
    concentrate in fraudulent transactions.
    """
    sql = text(
        """
        SELECT isFraud,
               COUNT(*)                                             AS n,
               AVG(balanceDeltaOrig)                                AS avg_delta_orig,
               AVG(ABS(balanceDeltaOrig))                            AS avg_abs_delta_orig,
               SUM(CASE WHEN ABS(balanceDeltaOrig) > 0.01 THEN 1 ELSE 0 END) AS n_nonzero_delta,
               AVG(origZeroedOut)                                     AS drain_rate
        FROM transactions
        GROUP BY isFraud;
        """
    )
    return pd.read_sql(sql, engine)


def get_transaction_overview_timeseries(engine: Engine) -> pd.DataFrame:
    """Per-step (hourly) transaction count and total volume, for the Transaction Overview dashboard view."""
    sql = text(
        """
        SELECT step,
               COUNT(*)     AS txn_count,
               SUM(amount)   AS total_amount
        FROM transactions
        GROUP BY step
        ORDER BY step;
        """
    )
    return pd.read_sql(sql, engine)


def get_model_run_history(engine: Engine) -> pd.DataFrame:
    """All model_runs rows, most recent first, for the Model Performance dashboard trend chart."""
    sql = text("SELECT * FROM model_runs ORDER BY run_id DESC;")
    return pd.read_sql(sql, engine)


def get_latest_deployed_run(engine: Engine) -> pd.Series | None:
    """The most recently deployed model_runs row, or None if no model has been trained yet."""
    df = pd.read_sql(
        text("SELECT * FROM model_runs WHERE is_deployed = 1 ORDER BY run_id DESC LIMIT 1;"),
        engine,
    )
    if df.empty:
        df = pd.read_sql(text("SELECT * FROM model_runs ORDER BY run_id DESC LIMIT 1;"), engine)
    return df.iloc[0] if not df.empty else None


def get_drift_summary(engine: Engine, run_id: int | None = None) -> pd.DataFrame:
    """
    Drift metrics for a given model run (or the latest run's drift metrics
    if run_id is None), for the Drift Monitoring dashboard view.
    """
    if run_id is None:
        sql = text(
            "SELECT * FROM drift_metrics WHERE run_id = (SELECT MAX(run_id) FROM drift_metrics) "
            "ORDER BY feature_name;"
        )
        return pd.read_sql(sql, engine)
    sql = text("SELECT * FROM drift_metrics WHERE run_id = :run_id ORDER BY feature_name;")
    return pd.read_sql(sql, engine, params={"run_id": run_id})


def get_system_health(engine: Engine) -> dict:
    """Row counts and last-run timestamps for the System Health dashboard view."""
    with engine.connect() as conn:
        n_transactions = conn.execute(text("SELECT COUNT(*) FROM transactions;")).scalar_one()
        n_fraud = conn.execute(text("SELECT SUM(isFraud) FROM transactions;")).scalar_one() or 0
        n_model_runs = conn.execute(text("SELECT COUNT(*) FROM model_runs;")).scalar_one()
        last_trained_at = conn.execute(
            text("SELECT trained_at FROM model_runs ORDER BY run_id DESC LIMIT 1;")
        ).scalar_one_or_none()
        last_retrain = conn.execute(
            text(
                "SELECT checked_at, triggered, trigger_reason FROM retrain_log "
                "ORDER BY id DESC LIMIT 1;"
            )
        ).fetchone()

    return {
        "n_transactions": n_transactions,
        "n_fraud": int(n_fraud),
        "n_model_runs": n_model_runs,
        "last_trained_at": last_trained_at,
        "last_retrain_check_at": last_retrain[0] if last_retrain else None,
        "last_retrain_triggered": bool(last_retrain[1]) if last_retrain else None,
        "last_retrain_reason": last_retrain[2] if last_retrain else None,
    }


def main():
    """CLI entrypoint: backfills velocity features against database/database.db."""
    engine = get_engine()
    print("Computing txnVelocity1h / txnVelocity24h and backfilling transactions...")
    n = get_velocity_features(engine)
    print(f"Backfilled velocity features for {n:,} rows.")


if __name__ == "__main__":
    main()
```

## Running it

```bash
python sql/queries.py
```

Requires `database/database.db` to already contain the `transactions`
table populated by `database/load_data.py`. This backfills
`txnVelocity1h`/`txnVelocity24h` for every row and is a prerequisite for
`models/train.py`, since `txnVelocity1h`/`txnVelocity24h` are part of
`MODEL_FEATURE_COLUMNS`.

## Query reference

| Function | Used by |
|---|---|
| `get_type_aggregates` | Dashboard — Transaction Overview |
| `get_fraud_breakdown` | Dashboard — Fraud Analytics |
| `get_cohort_balance_consistency` | Dashboard — Fraud Analytics |
| `get_transaction_overview_timeseries` | Dashboard — Transaction Overview |
| `get_model_run_history` | Dashboard — Model Performance |
| `get_drift_summary` | Dashboard — Drift Monitoring |
| `get_latest_deployed_run` | `drift/retrain.py` (baseline PR-AUC lookup) |
| `get_system_health` | Dashboard — System Health |

---
[← 04 — Database](04-database.md) | [Index](README.md) | Next: [06 — Modeling →](06-modeling.md)
