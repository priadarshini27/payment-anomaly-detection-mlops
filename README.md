# Payment Transaction Anomaly Detection with Data Drift Monitoring

> A production-inspired Machine Learning project that detects anomalous payment transactions using the **PaySim** dataset and continuously monitors **data drift** to ensure long-term model reliability.

Unlike traditional fraud detection projects that stop after model training, this project demonstrates the **complete Machine Learning lifecycle**, including data ingestion, SQL integration, anomaly detection, model evaluation, drift monitoring, retraining, and interactive visualization.

---

## 📥 Clone Repository

### HTTPS

```bash
git clone https://github.com/priadarshini27/payment-anomaly-detection-mlops.git
```

### Navigate to the Project Directory

```bash
cd payment-anomaly-detection-mlops
```


---

## 🚀 Project Objectives

- Detect anomalous payment transactions using **Isolation Forest**
- Store and analyze transaction data using **SQLite** and **SQL**
- Build a production-style ML pipeline
- Monitor feature distribution changes using:
  - Population Stability Index (PSI)
  - Kolmogorov–Smirnov (KS) Test
- Track model performance degradation
- Retrain the model when drift is detected
- Visualize the complete workflow using **Streamlit** and **Plotly**

---

## 📚 Documentation

The full documentation library — architecture, dataset/EDA findings, every
source file with exact paths, database schema, modeling and drift-monitoring
rationale, dashboard reference, testing, deployment, and troubleshooting —
lives in [`docs/`](docs/README.md). It is written so the entire project can
be recreated from these documents alone, without access to this repository.

Start at [`docs/01-getting-started.md`](docs/01-getting-started.md) for
environment setup and the full pipeline run sequence.

---

## ▶️ Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Place the PaySim dataset at data/paysim.csv (see data/README.md), then:
python preprocessing/clean_data.py   # clean, engineer features, load into SQLite
python sql/queries.py                # backfill velocity features
python models/train.py               # fit the Isolation Forest on the earliest 80% of rows
python models/evaluate.py            # evaluate on held-out rows + train/test overfitting check
python drift/retrain.py              # run a drift check, retrain if triggered
streamlit run dashboard/app.py       # launch the dashboard
```

Run `python -m pytest -q` to run the test suite.