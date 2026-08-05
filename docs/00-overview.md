# 00 — Project Overview

[← Back to index](README.md) | Next: [01 — Getting Started →](01-getting-started.md)

## What this project is

A production-inspired machine learning system that detects anomalous /
fraudulent payment transactions in the **PaySim** mobile-money dataset
using an **unsupervised Isolation Forest** model, and continuously
monitors feature drift to trigger automated retraining. It covers the
full ML lifecycle: data ingestion, SQL analytics, feature engineering,
model training, model evaluation, drift detection, automated retraining,
and an interactive dashboard.

Unlike a typical "train a classifier on a Kaggle CSV" project, this one
deliberately does **not** stop after model training. The reasons for
every major decision are documented as they occur, because in this
project the "why" matters as much as the "what" — see
[02-dataset-and-eda.md](02-dataset-and-eda.md) for the data-analysis
reasoning and [06-modeling.md](06-modeling.md) /
[08-drift-monitoring.md](08-drift-monitoring.md) for the modeling and
monitoring reasoning.

## Project objectives

- Detect anomalous/fraudulent payment transactions using an **unsupervised
  Isolation Forest**, trained without ever using the `isFraud` label as a
  feature.
- Persist and query transaction data through a normalized **SQLite**
  schema, using SQL for feature aggregation and cohort analysis.
- Build a production-style, **modular ML pipeline** separating ingestion,
  feature engineering, scoring, evaluation, and monitoring concerns.
- Quantify feature-level distribution shift over time using the
  **Population Stability Index (PSI)** and the two-sample
  **Kolmogorov–Smirnov (KS) test**.
- Track model performance degradation and **trigger retraining
  automatically** when drift or performance thresholds are breached.
- Visualize the end-to-end workflow through an interactive
  **Streamlit + Plotly** dashboard.

## System architecture

The system is organized into eight sequential stages. Each is
independently runnable so any stage can be re-executed in isolation (for
example, re-running drift monitoring without retraining the model).

```
┌─────────────────┐
│  PaySim CSV     │
└───────┬─────────┘
        ▼
┌─────────────────┐   chunked load (500k rows/batch),
│ Data Ingestion  │   dtype casting, schema validation
└───────┬─────────┘   (preprocessing/clean_data.py)
        ▼
┌─────────────────┐   duplicates, missing values, invalid
│ Data Cleaning   │   amounts removed; balances validated
└───────┬─────────┘   (preprocessing/clean_data.py)
        ▼
┌─────────────────┐   balanceDelta, origZeroedOut, logAmount,
│Feature Engineer │   isMerchantDest, type one-hot
└───────┬─────────┘   (preprocessing/feature_engineering.py)
        ▼
┌─────────────────┐   transactions, model_runs,
│ SQLite Store    │   drift_metrics, retrain_log tables
└───────┬─────────┘   (database/schema.sql, load_data.py)
        ▼
┌─────────────────┐   velocity features (window functions),
│  SQL Analytics  │   fraud/type/cohort aggregates
└───────┬─────────┘   (sql/queries.py)
        ▼
┌─────────────────┐   contamination-tuned, 200 estimators
│ Isolation Forest│   (models/train.py)
└───────┬─────────┘
        ▼
┌─────────────────┐   anomaly_score, threshold → binary flag
│ Anomaly Scoring │   (models/predict.py)
└───────┬─────────┘
        ▼
┌─────────────────┐   PR-AUC, ROC-AUC, precision@k, F1
│Model Evaluation │   (models/evaluate.py)
└───────┬─────────┘
        ▼
┌─────────────────┐   PSI + KS per feature, sliding
│ Drift Monitoring│   reference window (drift/psi.py, ks_test.py)
└───────┬─────────┘
     drift? ──► yes ──► Retrain trigger (drift/retrain.py) ─────┐
        │ no                                                    │
        ▼                                                       │
┌────────────────────────────┐  ◄───────────────────────────────┘
│Streamlit + Plotly Dashboard│
└────────────────────────────┘   (dashboard/app.py)
```

Drift monitoring closes the loop back into retraining, making the system
self-correcting rather than a one-shot batch job.

## Layer responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion & Cleaning | `preprocessing/clean_data.py` | Chunked CSV read (500k-row batches), schema validation, dtype coercion, cleaning rules |
| Feature Engineering | `preprocessing/feature_engineering.py` | Derives balance-delta errors, `origZeroedOut`, `logAmount`, `isMerchantDest`, one-hot `type` |
| Storage | `database/schema.sql`, `database/load_data.py` | Normalized tables: `transactions`, `model_runs`, `drift_metrics`, `retrain_log`; indexed on `step`, `nameOrig`, `type`, `isFraud` |
| Analytics | `sql/queries.py` | Parameterized SQL for velocity features, fraud breakdowns, cohort/balance-consistency checks |
| Modeling | `models/train.py` | Fits `sklearn.IsolationForest`, persists model + scaler via `joblib`, records hyperparameters |
| Scoring | `models/predict.py` | Loads a persisted model, scores transactions, writes `anomaly_score` / `is_anomaly` |
| Evaluation | `models/evaluate.py` | Computes PR-AUC, ROC-AUC, precision@k, recall@k, F1, confusion matrix by type |
| Drift | `drift/psi.py`, `drift/ks_test.py`, `drift/retrain.py` | Computes PSI and KS statistics per feature; applies the dual retraining trigger |
| Dashboard | `dashboard/app.py` | Streamlit app rendering six views backed by live SQLite queries and Plotly charts |

## Tech stack

| Category | Technology |
|---|---|
| Language | Python 3.11+ (developed and verified on Python 3.14) |
| Dataset | PaySim (synthetic mobile-money transactions) |
| Database | SQLite |
| Query Language | SQL (via SQLAlchemy) |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (`IsolationForest`, `StandardScaler`) |
| Statistics | SciPy (`ks_2samp`) |
| Drift Detection | PSI (custom implementation), KS Test (SciPy) |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Model Persistence | joblib |
| Testing | pytest |
| Version Control | Git & GitHub |

## Project structure

```
payment-anomaly-detection-mlops/
├── data/
│   ├── paysim.csv              # raw dataset (download separately, see 01)
│   ├── paysim_clean.csv        # produced by preprocessing/clean_data.py
│   └── README.md
├── database/
│   ├── schema.sql
│   ├── database.db             # produced by database/load_data.py
│   ├── load_data.py
│   └── __init__.py
├── preprocessing/
│   ├── clean_data.py
│   ├── feature_engineering.py
│   └── __init__.py
├── sql/
│   ├── queries.py
│   └── __init__.py
├── models/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── artifacts/              # produced by models/train.py (model_*.joblib, etc.)
│   └── __init__.py
├── drift/
│   ├── psi.py
│   ├── ks_test.py
│   ├── retrain.py
│   └── __init__.py
├── dashboard/
│   ├── app.py
│   └── __init__.py
├── eda/
│   └── eda.ipynb                # exploratory notebook, see 02
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_clean_data.py
│   ├── test_drift.py
│   └── __init__.py
├── docs/                          # this documentation library
├── doc/                            # original Word-document specification
├── pytest.ini
├── requirements.txt
└── README.md
```

Every one of these files is fully specified — path, purpose, and complete
source — somewhere in this documentation library.

---
[← Back to index](README.md) | Next: [01 — Getting Started →](01-getting-started.md)
