# 01 — Getting Started

[← 00 — Overview](00-overview.md) | [Index](README.md) | Next: [02 — Dataset & EDA →](02-dataset-and-eda.md)

This document takes you from an empty directory to a fully populated,
trained, and dashboard-ready project. Follow the steps in order.

## Prerequisites

- **Python 3.11+** (this project was built and verified on Python 3.14).
- **~2 GB free disk space** — the raw PaySim CSV is ~490 MB, the cleaned
  CSV is a similar size, and `database.db` grows to roughly 1.3 GB once
  fully loaded.
- **~4 GB free RAM** recommended for training (the full feature matrix for
  6.36M rows is held in memory during `models/train.py`).
- Git (to clone/initialize the repository).

## Step 1 — Create the project structure

### PowerShell

```powershell
mkdir data
mkdir database
mkdir preprocessing
mkdir sql
mkdir models
mkdir models\artifacts
mkdir drift
mkdir dashboard
mkdir eda
mkdir tests
mkdir docs
New-Item database\schema.sql -ItemType File
New-Item database\database.db -ItemType File
New-Item database\load_data.py -ItemType File
New-Item database\__init__.py -ItemType File
New-Item preprocessing\clean_data.py -ItemType File
New-Item preprocessing\feature_engineering.py -ItemType File
New-Item preprocessing\__init__.py -ItemType File
New-Item sql\queries.py -ItemType File
New-Item sql\__init__.py -ItemType File
New-Item models\train.py -ItemType File
New-Item models\predict.py -ItemType File
New-Item models\evaluate.py -ItemType File
New-Item models\__init__.py -ItemType File
New-Item drift\psi.py -ItemType File
New-Item drift\ks_test.py -ItemType File
New-Item drift\retrain.py -ItemType File
New-Item drift\__init__.py -ItemType File
New-Item dashboard\app.py -ItemType File
New-Item dashboard\__init__.py -ItemType File
New-Item tests\__init__.py -ItemType File
New-Item tests\test_feature_engineering.py -ItemType File
New-Item tests\test_clean_data.py -ItemType File
New-Item tests\test_drift.py -ItemType File
New-Item tests\test_train_split.py -ItemType File
New-Item pytest.ini -ItemType File
New-Item requirements.txt -ItemType File
New-Item README.md -ItemType File
```

### macOS / Linux / Git Bash

```bash
mkdir -p data database preprocessing sql models/artifacts drift dashboard eda tests docs
touch database/schema.sql database/database.db database/load_data.py database/__init__.py
touch preprocessing/clean_data.py preprocessing/feature_engineering.py preprocessing/__init__.py
touch sql/queries.py sql/__init__.py
touch models/train.py models/predict.py models/evaluate.py models/__init__.py
touch drift/psi.py drift/ks_test.py drift/retrain.py drift/__init__.py
touch dashboard/app.py dashboard/__init__.py
touch tests/__init__.py tests/test_feature_engineering.py tests/test_clean_data.py tests/test_drift.py tests/test_train_split.py
touch pytest.ini requirements.txt README.md
```

Now paste the source code from each corresponding document in this
library into the file at that exact path:

| File | Source in this library |
|---|---|
| `preprocessing/feature_engineering.py` | [03-data-cleaning.md](03-data-cleaning.md) |
| `preprocessing/clean_data.py` | [03-data-cleaning.md](03-data-cleaning.md) |
| `database/schema.sql` | [04-database.md](04-database.md) |
| `database/load_data.py` | [04-database.md](04-database.md) |
| `sql/queries.py` | [05-sql-analytics.md](05-sql-analytics.md) |
| `models/train.py`, `predict.py`, `evaluate.py` | [06-modeling.md](06-modeling.md), [07-evaluation-metrics.md](07-evaluation-metrics.md) |
| `drift/psi.py`, `ks_test.py`, `retrain.py` | [08-drift-monitoring.md](08-drift-monitoring.md) |
| `dashboard/app.py` | [09-dashboard.md](09-dashboard.md) |
| `tests/*.py`, `pytest.ini` | [10-testing-and-validation.md](10-testing-and-validation.md) |

The `__init__.py` files are all **empty** — they exist only so Python
treats `preprocessing/`, `database/`, `sql/`, `models/`, `drift/`,
`dashboard/`, and `tests/` as importable packages, which the cross-module
imports (e.g. `from sql.queries import get_engine` inside
`models/train.py`) depend on.

## Step 2 — Create a virtual environment and install dependencies

### `requirements.txt`

```text
pandas
numpy
scikit-learn
scipy
matplotlib
plotly
streamlit
joblib
sqlalchemy
jupyter
ipykernel
pytest
```

### PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3 — Download the dataset

1. Download the PaySim dataset from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1
2. Extract the downloaded archive. The CSV inside is typically named
   something like `PS_20174392719_1491204439457_log.csv`.
3. Rename it to `paysim.csv` and place it at `data/paysim.csv`.

The full dataset is **6,362,620 rows** and roughly **490 MB**. Every
pipeline stage in this project is written to run against the full file —
see [02-dataset-and-eda.md](02-dataset-and-eda.md) for the schema and
class-imbalance details that shape the rest of the design.

## Step 4 — Run the full pipeline end-to-end

Run these commands from the project root, in this exact order — each
stage depends on artifacts produced by the previous one.

```bash
# 1. Clean the raw CSV, engineer features, write data/paysim_clean.csv,
#    and load the result into database/database.db.
python preprocessing/clean_data.py

# 2. (Only needed if you ran clean_data.py with --skip-db, or want to
#    reload without re-cleaning.) Load data/paysim_clean.csv into SQLite.
python database/load_data.py

# 3. Backfill txnVelocity1h / txnVelocity24h into the transactions table
#    (requires a window-function query over the full ordered history).
python sql/queries.py

# 4. Fit the Isolation Forest on the earliest 80% of rows (by step) and
#    persist model + scaler + reference window artifacts. Later rows are
#    held out for evaluation.
python models/train.py

# 5. Evaluate on the held-out rows against isFraud (PR-AUC, ROC-AUC,
#    precision@k, recall@k, F1, confusion matrix), and compare with the
#    training rows to check for overfitting.
python models/evaluate.py

# 6. Run a drift check comparing the training reference window against a
#    recent scoring window, and retrain automatically if triggered.
python drift/retrain.py

# 7. Launch the dashboard.
streamlit run dashboard/app.py
```

On the full 6.36M-row dataset, expect step 1 (cleaning) to take roughly
2.5–3 minutes, step 2 (SQLite load) a similar amount of time, step 4
(training) about 3 minutes, and step 5 (evaluation) about 6 minutes.
Fitting the forest itself is fast, because scikit-learn's
`IsolationForest` subsamples 256 rows per tree by default
(`max_samples='auto'`) regardless of the dataset's size; training time is
mostly spent loading the feature matrix from SQLite (see
[06-modeling.md](06-modeling.md)). Evaluation scores every row, both the
held-out rows and the training rows used for the overfitting check, with
all 200 trees.

## Step 5 — Verify

```bash
python -m pytest -q
```

All tests should pass. See
[10-testing-and-validation.md](10-testing-and-validation.md) for the full
test source and a manual validation checklist (expected row counts,
metric sanity ranges) you can run against your own populated database.

---
[← 00 — Overview](00-overview.md) | [Index](README.md) | Next: [02 — Dataset & EDA →](02-dataset-and-eda.md)
