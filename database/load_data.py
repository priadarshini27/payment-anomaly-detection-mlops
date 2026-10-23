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
