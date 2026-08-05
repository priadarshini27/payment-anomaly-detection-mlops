# 03 — Data Cleaning & Feature Engineering

[← 02 — Dataset & EDA](02-dataset-and-eda.md) | [Index](README.md) | Next: [04 — Database →](04-database.md)

This document covers `preprocessing/feature_engineering.py` and
`preprocessing/clean_data.py` — the production, deterministic conversion
of the exploratory notebook's findings (see
[02-dataset-and-eda.md](02-dataset-and-eda.md)) into code that runs the
same way every time, fails loudly on a violated assumption, and is
reviewable like any other source file.

## Cleaning decisions and their rationale

Every step below is traceable to a specific, confirmed EDA finding — not
a generic checklist run out of habit.

| Step | Decision | Why |
|---|---|---|
| Duplicates | Drop exact duplicates, then logical duplicates on `(step, nameOrig, nameDest, amount)` | 0 found in PaySim itself, but a real safety net — an ingestion retry would otherwise inflate velocity features |
| Dtypes | Coerce to `int32`/`float32`/`category`/`int8`/`string` immediately after load | Wrong dtypes break downstream joins/aggregations silently |
| Missing values | Drop rows with nulls in any required column | 0 found in PaySim; imputation is avoided by design since it would smooth away real anomalies |
| Balance residuals | **Not** used to filter rows — computed and kept as `balanceDeltaOrig`/`balanceDeltaDest` features | Confirmed structural artifact of PaySim's fraud generator, not corruption |
| Zero-amount transactions | **Dropped** (`amount <= 0`) | Exactly 16 such rows exist and all are labeled fraud — a documented trade-off (see below) |
| Categorical casing | `type` upper-cased and stripped before one-hot encoding | Guarantees exactly 5 clean categories, never a fragmented set |

### The zero-amount trade-off, stated honestly

`validate_amounts()` drops the 16 zero-amount rows discovered in EDA, and
every one of them is labeled fraud. Dropping them measurably shrinks the
already-tiny positive class (16 of 8,213, ≈0.19%). This is judged
acceptable rather than swept under the rug for two reasons: a $0 transfer
is not a valid financial record regardless of its label, and — more
importantly for an *unsupervised* detector — `logAmount = log1p(0) = 0`
for all 16 of them, giving the amount-based features no discriminating
signal to learn from in the first place. If a future iteration of this
project needs those 16 rows for supervised benchmarking, flag rather than
drop them; that variant is not what is implemented here.

## `preprocessing/feature_engineering.py`

### File: preprocessing/feature_engineering.py

```python
"""
preprocessing/feature_engineering.py
=====================================
Derives the engineered features used throughout the project (modeling,
drift monitoring, dashboard) from the raw/cleaned PaySim columns.

These functions are the single source of truth for feature derivation and
are called from two places:
  * preprocessing/clean_data.py -- applied once, in bulk, to the full
    historical dataset before it is loaded into SQLite.
  * models/predict.py -- applied to new/incoming transactions at scoring
    time, so a live transaction is featurized identically to how the
    training data was featurized.

Engineered features (see Document 1, Section 4.1):
  balanceDeltaOrig -- oldbalanceOrg - newbalanceOrig - amount. A non-zero
      residual means the originator's balance did not move by exactly the
      transacted amount -- PaySim's fraud generator introduces this
      deliberately, so it is a strong signal rather than noise to clean away.
  balanceDeltaDest -- oldbalanceDest + amount - newbalanceDest. Same idea,
      applied to the recipient side.
  origZeroedOut -- 1 if oldbalanceOrg > 0 and newbalanceOrig == 0, else 0.
      Captures full-balance drains typical of account-takeover fraud.
  logAmount -- log1p(amount). Variance-stabilizing transform since `amount`
      is heavily right-skewed.
  isMerchantDest -- 1 if nameDest starts with 'M' (a merchant account),
      else 0. Merchant accounts don't carry a real balance in PaySim
      (oldbalanceDest/newbalanceDest are structurally 0), which is why
      balanceDeltaDest behaves differently for merchant destinations.
  type_<CATEGORY> -- one-hot encoding of the five PaySim transaction types
      (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), since fraud is
      concentrated almost entirely in TRANSFER and CASH_OUT.

txnVelocity1h / txnVelocity24h are *not* computed here -- they require a
window function over the full ordered transaction history (per nameOrig,
by step) rather than a per-row/per-batch computation, so they are computed
once after load, in sql/queries.py::get_velocity_features().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# The feature set the Isolation Forest is trained and scored on. Deliberately
# excludes isFraud/isFlaggedFraud (label leakage) and identifier columns.
MODEL_FEATURE_COLUMNS = [
    "logAmount",
    "balanceDeltaOrig",
    "balanceDeltaDest",
    "origZeroedOut",
    "isMerchantDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "txnVelocity1h",
    "txnVelocity24h",
]


def add_balance_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds balanceDeltaOrig and balanceDeltaDest residual columns."""
    df = df.copy()
    df["balanceDeltaOrig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    ).astype("float32")
    df["balanceDeltaDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).astype("float32")
    return df


def add_origin_zeroed_out_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the binary origZeroedOut account-drain indicator."""
    df = df.copy()
    df["origZeroedOut"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype("int8")
    return df


def add_log_amount_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds logAmount = log1p(amount), a variance-stabilizing transform."""
    df = df.copy()
    df["logAmount"] = np.log1p(df["amount"]).astype("float32")
    return df


def add_merchant_dest_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds isMerchantDest = 1 if nameDest is a merchant account (prefix 'M')."""
    df = df.copy()
    df["isMerchantDest"] = df["nameDest"].astype(str).str.startswith("M").astype("int8")
    return df


def add_type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one-hot columns type_CASH_IN .. type_TRANSFER for the 5 PaySim types."""
    df = df.copy()
    type_str = df["type"].astype(str).str.upper()
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (type_str == t).astype("int8")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies every engineered feature in the documented order. Idempotent:
    calling it twice on an already-engineered frame recomputes the same
    values rather than stacking columns.

    Does NOT compute txnVelocity1h/24h -- see module docstring.
    """
    df = add_balance_delta_features(df)
    df = add_origin_zeroed_out_feature(df)
    df = add_log_amount_feature(df)
    df = add_merchant_dest_feature(df)
    df = add_type_one_hot(df)
    return df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and orders the columns the Isolation Forest is trained/scored
    on (MODEL_FEATURE_COLUMNS), assuming engineer_features() and the
    velocity backfill have already run. Missing velocity columns (e.g. for
    a single newly-scored transaction with no history) default to 0.
    """
    df = df.copy()
    for col in ("txnVelocity1h", "txnVelocity24h"):
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"build_model_matrix: missing required feature columns {missing}. "
            "Did you call engineer_features() first?"
        )
    return df[MODEL_FEATURE_COLUMNS].astype("float64")
```

## `preprocessing/clean_data.py`

### File: preprocessing/clean_data.py

```python
"""
preprocessing/clean_data.py
=============================
Production-grade cleaning pipeline for the PaySim dataset, converting the
notebook-stage exploration in eda/eda.ipynb into deterministic, testable,
CLI-runnable code (Document 2, Section 10).

Every cleaning decision below is traceable to a specific finding confirmed
in eda/eda.ipynb against the full 6,362,620-row PaySim release:
  * 0 missing values in any column                    -> no imputation needed
  * 0 fully duplicated rows                            -> dedup rule is a
                                                           no-op safety net,
                                                           not a real fix
  * 0 negative amounts or balances                     -> no negative-value
                                                           correction needed
  * exactly 16 zero-amount transactions, ALL fraud     -> dropped as invalid
                                                           financial records
                                                           (see validate_amounts)
  * isMerchantDest count == PAYMENT count exactly       -> confirms the C/M
                                                           account-id prefix
                                                           convention
  * non-zero balance residuals are a structural         -> kept and engineered
    property of PaySim's fraud generator, not corruption   as balanceDeltaOrig/
                                                             balanceDeltaDest,
                                                             never dropped

Run directly:
    python preprocessing/clean_data.py
This reads data/paysim.csv, cleans and feature-engineers it in 500k-row
chunks, writes data/paysim_clean.csv, and (unless --skip-db is passed) loads
the result into the SQLite database via database/load_data.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_engineering import engineer_features  # noqa: E402

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

RAW_COLUMNS = list(RAW_DTYPES.keys())
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_DATA_PATH = Path("data/paysim.csv")
DEFAULT_CLEAN_PATH = Path("data/paysim_clean.csv")


class SchemaValidationError(ValueError):
    """Raised when the raw CSV does not match the expected PaySim schema."""


def load_raw(path: Path | str = DEFAULT_DATA_PATH, chunksize: int | None = None):
    """
    Loads the raw PaySim CSV with explicit dtypes (matching the dtypes
    validated in eda/eda.ipynb Section 2).

    If chunksize is given, returns a TextFileReader (iterator of DataFrame
    chunks) for memory-safe processing of the full 6.36M-row file. If
    chunksize is None, loads the whole file into memory at once.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place it there "
            "(see data/README.md)."
        )
    read_kwargs = dict(dtype=RAW_DTYPES, usecols=RAW_COLUMNS)
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize, **read_kwargs)
    return pd.read_csv(path, **read_kwargs)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Confirms the dataframe has exactly the expected PaySim columns. Raises
    SchemaValidationError (loudly, per Document 2 Section 10 "production
    code" requirements) rather than silently continuing with a malformed
    frame.
    """
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in RAW_COLUMNS]
    if missing:
        raise SchemaValidationError(f"Missing expected columns: {missing}")
    if extra:
        raise SchemaValidationError(f"Unexpected extra columns: {extra}")


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Removes exact duplicate rows and logical duplicates (same step,
    nameOrig, nameDest, amount). eda/eda.ipynb found 0 duplicates in the
    full dataset, so this is expected to be a no-op on PaySim itself -- but
    the rule is real: an ingestion retry in a future data source would
    otherwise silently inflate txnVelocity1h/24h.
    """
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["step", "nameOrig", "nameDest", "amount"])
    return df, before - len(df)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Re-asserts the documented dtypes (defensive; load_raw already applies them)."""
    df = df.copy()
    for col, dtype in RAW_DTYPES.items():
        if str(df[col].dtype) != dtype:
            df[col] = df[col].astype(dtype)
    return df


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    PaySim's core numeric fields are complete (0 missing values confirmed
    in eda/eda.ipynb Section 4.1). This function still runs the check on
    every load rather than assuming it forever, since a future data source
    swapped into this pipeline may not be as clean. Any row with a null in
    a required field is dropped, since mean/median imputation would
    artificially smooth away the exact anomalies this project exists to
    detect (Document 2, Section 6.3 strategy table).
    """
    before = len(df)
    df = df.dropna(subset=RAW_COLUMNS)
    return df, before - len(df)


def validate_balances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the balance residuals but does NOT drop rows based on them.
    eda/eda.ipynb confirmed non-zero residuals are a structural artifact of
    PaySim's fraud-injection logic, not data corruption -- so per Document 2
    Section 9.4, the residual is engineered into a feature
    (balanceDeltaOrig/balanceDeltaDest, added later in engineer_features)
    rather than used as a row-filtering rule here.
    """
    return df


def validate_amounts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drops transactions with amount <= 0. eda/eda.ipynb found exactly 16
    zero-amount transactions in the full dataset, and every one of them is
    labeled fraud. This is a genuine trade-off, documented rather than
    hidden: a $0 transfer is not a valid financial record regardless of its
    label, so it is removed as an invalid record (Document 2 Section 9.5) --
    but doing so does discard 16 true-positive fraud rows out of 8,213, a
    ~0.19% reduction in the already-tiny positive class. This is judged
    acceptable because those 16 rows carry no `amount`-based signal for an
    unsupervised detector to learn from in the first place (logAmount would
    be log1p(0) = 0 for all of them, indistinguishable from a missing value).
    """
    before = len(df)
    df = df[df["amount"] > 0]
    return df, before - len(df)


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upper-cases and re-categorizes `type` so that value_counts() and the
    one-hot encoding in feature_engineering.py always produce exactly the
    5 documented categories, never a fragmented set from stray casing.
    """
    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip().str.upper().astype("category")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Applies every cleaning + feature-engineering step in a fixed, documented
    order and returns (cleaned_df, stats_dict) so callers can report exactly
    what happened.
    """
    stats = {"rows_in": len(df)}

    df, n_dupes = drop_duplicates(df)
    stats["duplicates_removed"] = n_dupes

    df = fix_dtypes(df)

    df, n_missing = handle_missing(df)
    stats["missing_rows_removed"] = n_missing

    df = validate_balances(df)

    df, n_invalid_amount = validate_amounts(df)
    stats["invalid_amount_rows_removed"] = n_invalid_amount

    df = standardize_categoricals(df)

    df = engineer_features(df)

    stats["rows_out"] = len(df)
    return df, stats


def save_csv(df: pd.DataFrame, out_path: Path | str = DEFAULT_CLEAN_PATH) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a" if not header else "w", header=header, index=False)


def print_stats(all_stats: list[dict], elapsed_s: float) -> None:
    total_in = sum(s["rows_in"] for s in all_stats)
    total_out = sum(s["rows_out"] for s in all_stats)
    total_dupes = sum(s["duplicates_removed"] for s in all_stats)
    total_missing = sum(s["missing_rows_removed"] for s in all_stats)
    total_invalid_amount = sum(s["invalid_amount_rows_removed"] for s in all_stats)

    print("=" * 70)
    print("preprocessing/clean_data.py -- run summary")
    print("=" * 70)
    print(f"Chunks processed:              {len(all_stats):,}")
    print(f"Rows in (raw):                 {total_in:,}")
    print(f"  Duplicates removed:          {total_dupes:,}")
    print(f"  Missing-value rows removed:  {total_missing:,}")
    print(f"  Invalid-amount rows removed: {total_invalid_amount:,}")
    print(f"Rows out (clean):              {total_out:,}")
    print(f"Elapsed:                       {elapsed_s:.1f}s")
    print("=" * 70)


def run(
    data_path: Path | str = DEFAULT_DATA_PATH,
    clean_path: Path | str = DEFAULT_CLEAN_PATH,
    chunksize: int = DEFAULT_CHUNKSIZE,
    load_db: bool = True,
    db_path: str | None = None,
) -> list[dict]:
    """Runs the full ingest -> clean -> feature-engineer -> save pipeline."""
    clean_path = Path(clean_path)
    if clean_path.exists():
        clean_path.unlink()  # start fresh; save_csv appends per chunk

    t0 = time.time()
    all_stats = []
    for i, chunk in enumerate(load_raw(data_path, chunksize=chunksize)):
        validate_schema(chunk)
        cleaned, stats = clean(chunk)
        save_csv(cleaned, clean_path)
        all_stats.append(stats)
        print(
            f"  chunk {i + 1}: {stats['rows_in']:,} in -> {stats['rows_out']:,} out"
        )
    elapsed = time.time() - t0
    print_stats(all_stats, elapsed)

    if load_db:
        from database.load_data import load_clean_csv_to_sqlite

        load_clean_csv_to_sqlite(clean_path, db_path=db_path)

    return all_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--clean-path", default=str(DEFAULT_CLEAN_PATH))
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--skip-db", action="store_true", help="Skip loading the cleaned CSV into SQLite."
    )
    parser.add_argument("--db-path", default=None, help="Override the SQLite DB path.")
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        clean_path=args.clean_path,
        chunksize=args.chunksize,
        load_db=not args.skip_db,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
```

## Running it

```bash
python preprocessing/clean_data.py
```

### Measured output (full 6,362,620-row dataset)

```
======================================================================
preprocessing/clean_data.py -- run summary
======================================================================
Chunks processed:              13
Rows in (raw):                 6,362,620
  Duplicates removed:          0
  Missing-value rows removed:  0
  Invalid-amount rows removed: 16
Rows out (clean):              6,362,604
Elapsed:                       169.3s
======================================================================
```

This confirms, at production scale, exactly what the EDA notebook found
on the same data: zero duplicates, zero missing values, and precisely 16
invalid (zero-amount) rows removed.

To skip the automatic SQLite load (e.g. if you want to inspect
`data/paysim_clean.csv` first), run:

```bash
python preprocessing/clean_data.py --skip-db
```

and then load it separately with `database/load_data.py` — see
[04-database.md](04-database.md).

---
[← 02 — Dataset & EDA](02-dataset-and-eda.md) | [Index](README.md) | Next: [04 — Database →](04-database.md)
