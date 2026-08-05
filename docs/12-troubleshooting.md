# 12 — Troubleshooting

[← 11 — Deployment & Operations](11-deployment-and-operations.md) | [Index](README.md)

## `FileNotFoundError: PaySim CSV not found at data/paysim.csv`

`preprocessing/clean_data.py::load_raw` raises this deliberately rather
than proceeding on missing data. Download the dataset from
https://www.kaggle.com/datasets/ealaxi/paysim1, extract it, rename the
CSV to `paysim.csv`, and place it at `data/paysim.csv` (see
[01-getting-started.md](01-getting-started.md), Step 3).

## `SchemaValidationError: Missing expected columns: [...]`

The CSV at `data/paysim.csv` doesn't have the 11 columns PaySim ships
with, in the expected names. This usually means a different file was
placed there, or the CSV was re-exported with different column names.
Compare the header row of your file against `RAW_COLUMNS` in
`preprocessing/clean_data.py`.

## `MemoryError` or the process is killed during `clean_data.py` / `load_data.py`

Both scripts process the CSV in chunks (`DEFAULT_CHUNKSIZE = 500_000` for
cleaning, `100_000` for loading) specifically to avoid holding the full
6.36M-row dataset in memory at once. On a memory-constrained machine,
lower the chunk size:

```bash
python preprocessing/clean_data.py --chunksize 100000
python database/load_data.py --chunksize 25000
```

## `sqlite3.OperationalError: database is locked`

SQLite serializes writers. This happens if two write jobs run
concurrently against the same `database.db` — e.g. `load_data.py` and
`train.py` at the same time, or the Streamlit dashboard mid-write (the
dashboard itself never writes, but a stray script pointed at the same
file might). Ensure pipeline stages run sequentially, not in parallel,
against a shared database file. See
[11-deployment-and-operations.md](11-deployment-and-operations.md) for
the broader concurrency note.

## `KeyError: ['txnVelocity1h', 'txnVelocity24h']` (or similar) from `build_model_matrix`

This means `engineer_features()` ran but the velocity backfill
(`sql/queries.py::get_velocity_features`) has not. Run:

```bash
python sql/queries.py
```

before `models/train.py`. Note that `build_model_matrix()` will actually
*not* raise for missing velocity columns specifically — it defaults them
to 0 — but every other required column (e.g. `logAmount`,
`balanceDeltaOrig`) will raise if `engineer_features()` was skipped
entirely. If you see this for a non-velocity column, the transaction data
was loaded without going through `preprocessing/clean_data.py`.

## `RuntimeError: No rows found in 'transactions'.` from `models/train.py`

The `transactions` table is empty. Run the full pipeline in order:
`clean_data.py` → `load_data.py` → `sql/queries.py` → `train.py`. Verify
with:

```bash
sqlite3 database/database.db "SELECT COUNT(*) FROM transactions;"
```

## `RuntimeError: No model_runs row found.` from `models/evaluate.py` or `drift/retrain.py`

No model has been trained yet. Run `python models/train.py` first — every
successful call inserts exactly one row into `model_runs`.

## Dtype mismatches / `df['type']` behaving unexpectedly

`type` is loaded as a pandas `category` dtype
(`RAW_DTYPES["type"] = "category"`) for memory efficiency on the full
6.36M-row file. If you're inspecting a chunk manually and see unexpected
behavior with string methods, cast explicitly:

```python
df["type"].astype(str)
```

`standardize_categoricals()` already does this internally before
re-categorizing, so this only matters for ad-hoc debugging, not for the
pipeline itself.

## Streamlit dashboard shows "The database has no transactions yet"

This is not a bug — it's `dashboard/app.py::_db_is_ready()` correctly
detecting an empty or missing `transactions` table and telling you
exactly which commands to run, in order, rather than crashing on the
first query. Follow the six commands it prints.

## `ModuleNotFoundError: No module named 'preprocessing'` (or `sql`, `models`, `drift`)

Every cross-module import in this project (e.g.
`from preprocessing.feature_engineering import engineer_features` inside
`preprocessing/clean_data.py`) relies on two things: (1) an empty
`__init__.py` in each package directory, and (2) the project root being
on `sys.path`. Every script that needs it inserts the project root itself
(`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`), so
this should only occur if you're running a script from somewhere other
than the project root, or if an `__init__.py` file is missing — verify
`preprocessing/__init__.py`, `database/__init__.py`, `sql/__init__.py`,
`models/__init__.py`, `drift/__init__.py`, and `dashboard/__init__.py`
all exist (they're empty, but they must exist).

## PSI or KS test returns 0 / p-value of 1.0 unexpectedly

Both `drift/psi.py::calculate_psi` and `drift/ks_test.py::ks_drift_test`
degrade gracefully to a "no drift" result (`psi=0.0`, `ks_pvalue=1.0`)
when either the reference or current window is empty or too small
(fewer than 2 points for KS). This is deliberate — it means a
misconfigured `--current-step-start`/`--current-step-end` range that
matches zero rows won't crash `drift/retrain.py`, but it will also
silently report "no drift" instead of erroring. `check_and_retrain()`
does raise a `ValueError` if the *current* window is entirely empty
before this can happen — but a window with very few rows will still
produce a valid-looking, low-confidence result. Sanity-check your
`step` range against `SELECT COUNT(*) FROM transactions WHERE step
BETWEEN ? AND ?` if drift results look suspiciously flat.

---
[← 11 — Deployment & Operations](11-deployment-and-operations.md) | [Index](README.md)
