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