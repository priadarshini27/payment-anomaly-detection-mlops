# 09 — Dashboard

[← 08 — Drift Monitoring](08-drift-monitoring.md) | [Index](README.md) | Next: [10 — Testing & Validation →](10-testing-and-validation.md)

`dashboard/app.py` is a Streamlit application rendering six views, each
backed by live SQLite queries (`sql/queries.py`) and charted with Plotly.

## The six views

| View | Contents |
|---|---|
| Transaction Overview | Volume and value trends by step and type; live SQL aggregates. |
| Fraud Analytics | Flagged-transaction breakdown by type, amount band, and time; precision/recall at the current threshold. |
| Transaction Analysis | Drill-down into individual transaction feature values and anomaly score. |
| Model Performance | PR-AUC / ROC-AUC trend across `model_runs` versions; confusion matrix. |
| Drift Monitoring | Per-feature PSI and KS p-value over time, with threshold bands overlaid. |
| System Health | Pipeline run status, last ingestion timestamp, row counts, last retrain trigger reason. |

## `dashboard/app.py`

### File: dashboard/app.py

```python
"""
dashboard/app.py
===================
Streamlit dashboard rendering the six views from Document 1 Section 7,
backed by live SQLite queries (sql/queries.py) and charted with Plotly.

Run:
    streamlit run dashboard/app.py

Views:
  1. Transaction Overview -- volume/value trends by step and type.
  2. Fraud Analytics       -- flagged-transaction breakdown by type, amount
                               band, and time; precision/recall at the
                               current threshold.
  3. Transaction Analysis   -- drill-down into individual transactions'
                               feature values and anomaly score.
  4. Model Performance       -- PR-AUC / ROC-AUC trend across model_runs
                               versions; confusion matrix.
  5. Drift Monitoring         -- per-feature PSI and KS p-value over time,
                               with threshold bands overlaid.
  6. System Health              -- pipeline run status, last ingestion
                               timestamp, row counts, last retrain reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql.queries import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_cohort_balance_consistency,
    get_drift_summary,
    get_engine,
    get_fraud_breakdown,
    get_model_run_history,
    get_system_health,
    get_transaction_overview_timeseries,
    get_type_aggregates,
)

st.set_page_config(page_title="Payment Anomaly Detection", layout="wide")

VIEWS = [
    "Transaction Overview",
    "Fraud Analytics",
    "Transaction Analysis",
    "Model Performance",
    "Drift Monitoring",
    "System Health",
]


@st.cache_resource
def _engine():
    return get_engine(DEFAULT_DB_PATH)


def _db_is_ready(engine) -> bool:
    try:
        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM transactions;")).scalar_one()
        return n > 0
    except Exception:
        return False


def view_transaction_overview(engine):
    st.header("Transaction Overview")
    ts = get_transaction_overview_timeseries(engine)
    types = get_type_aggregates(engine)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(ts, x="step", y="txn_count", title="Transaction Count by Step (hour)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(ts, x="step", y="total_amount", title="Transaction Volume by Step (hour)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume and Value by Transaction Type")
    st.dataframe(types, use_container_width=True)
    fig = px.bar(types, x="type", y="txn_count", title="Transaction Count by Type")
    st.plotly_chart(fig, use_container_width=True)


def view_fraud_analytics(engine):
    st.header("Fraud Analytics")
    breakdown = get_fraud_breakdown(engine)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            breakdown["by_type"], x="type", y="fraud_rate",
            title="Fraud Rate by Transaction Type", log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            breakdown["by_amount_band"], x="amount_band", y="fraud_rate",
            title="Fraud Rate by Amount Band",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Rate Over Time (per day)")
    fig = px.line(breakdown["by_step_bucket"], x="day", y="fraud_rate", title="Daily Fraud Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Balance-Drain Pattern by Fraud Label")
    st.dataframe(get_cohort_balance_consistency(engine), use_container_width=True)

    latest = get_model_run_history(engine)
    if not latest.empty and pd.notna(latest.iloc[0].get("precision_at_k")):
        row = latest.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@k (current threshold)", f"{row['precision_at_k']:.3f}")
        c2.metric("Recall@k (current threshold)", f"{row['recall_at_k']:.3f}")
        c3.metric("F1 @ threshold", f"{row['f1_at_threshold']:.3f}")


def view_transaction_analysis(engine):
    st.header("Transaction Analysis")
    st.caption("Drill down into individual transactions and their engineered features / anomaly score.")

    txn_id = st.number_input("Transaction ID", min_value=1, step=1, value=1)
    df = pd.read_sql(text("SELECT * FROM transactions WHERE txn_id = :id"), engine, params={"id": int(txn_id)})
    if df.empty:
        st.warning("No transaction found with that ID.")
    else:
        st.dataframe(df.T.rename(columns={0: "value"}), use_container_width=True)

    st.subheader("Top Anomalies (highest anomaly_score)")
    top = pd.read_sql(
        text(
            "SELECT txn_id, step, type, amount, isFraud, anomaly_score, is_anomaly "
            "FROM transactions WHERE anomaly_score IS NOT NULL "
            "ORDER BY anomaly_score DESC LIMIT 50;"
        ),
        engine,
    )
    st.dataframe(top, use_container_width=True)


def view_model_performance(engine):
    st.header("Model Performance")
    history = get_model_run_history(engine)
    if history.empty:
        st.info("No trained models found yet. Run models/train.py and models/evaluate.py.")
        return

    evaluated = history.dropna(subset=["pr_auc"])
    if not evaluated.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=evaluated["run_id"], y=evaluated["pr_auc"], name="PR-AUC", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=evaluated["run_id"], y=evaluated["roc_auc"], name="ROC-AUC", mode="lines+markers"))
        fig.update_layout(title="PR-AUC / ROC-AUC Trend Across Model Runs", xaxis_title="run_id", yaxis_title="score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("model_runs")
    st.dataframe(history, use_container_width=True)


def view_drift_monitoring(engine):
    st.header("Drift Monitoring")
    drift = get_drift_summary(engine)
    if drift.empty:
        st.info("No drift checks found yet. Run drift/retrain.py.")
        return

    fig = px.bar(
        drift, x="feature_name", y="psi_value", color="psi_flag",
        title="PSI by Feature (latest drift check)",
        color_discrete_map={"none": "#4c9a6b", "moderate": "#d1a23c", "significant": "#c0392b"},
    )
    fig.add_hline(y=0.10, line_dash="dot", annotation_text="moderate (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", annotation_text="significant (0.25)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(drift, x="feature_name", y="ks_pvalue", title="KS Test p-value by Feature")
    fig2.add_hline(y=0.05, line_dash="dash", annotation_text="alpha = 0.05")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("drift_metrics (latest run)")
    st.dataframe(drift, use_container_width=True)


def view_system_health(engine):
    st.header("System Health")
    health = get_system_health(engine)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{health['n_transactions']:,}")
    c2.metric("Total Fraud Transactions", f"{health['n_fraud']:,}")
    c3.metric("Model Versions Trained", health["n_model_runs"])

    st.subheader("Pipeline Status")
    st.write(f"**Last model trained at:** {health['last_trained_at'] or 'never'}")
    st.write(f"**Last drift check at:** {health['last_retrain_check_at'] or 'never'}")
    st.write(f"**Last retrain triggered:** {health['last_retrain_triggered']}")
    st.write(f"**Last retrain reason:** {health['last_retrain_reason'] or 'n/a'}")


def main():
    st.title("Payment Transaction Anomaly Detection -- Dashboard")
    engine = _engine()

    if not _db_is_ready(engine):
        st.error(
            "The database has no transactions yet. Run the pipeline first:\n\n"
            "1. python preprocessing/clean_data.py\n"
            "2. python database/load_data.py\n"
            "3. python sql/queries.py\n"
            "4. python models/train.py\n"
            "5. python models/evaluate.py\n"
            "6. python drift/retrain.py"
        )
        return

    view = st.sidebar.radio("View", VIEWS)

    if view == "Transaction Overview":
        view_transaction_overview(engine)
    elif view == "Fraud Analytics":
        view_fraud_analytics(engine)
    elif view == "Transaction Analysis":
        view_transaction_analysis(engine)
    elif view == "Model Performance":
        view_model_performance(engine)
    elif view == "Drift Monitoring":
        view_drift_monitoring(engine)
    elif view == "System Health":
        view_system_health(engine)


if __name__ == "__main__":
    main()
```

## Running it

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default. The dashboard reads directly
from `database/database.db` — it never writes to it, so it is always safe
to have it open while other pipeline stages run (though a heavy write,
like a full `database/load_data.py` reload, will briefly lock SQLite; the
dashboard will simply show a transient error on the next query and
recover once the write finishes).

If the database has no transactions yet, the app shows the exact pipeline
commands needed to populate it, rather than crashing on an empty table.

## Reading the Model Performance trend

The PR-AUC/ROC-AUC trend plots each run's stored `pr_auc`/`roc_auc`, and
those don't all mean the same thing:

- **Run 1** was trained and evaluated on all rows, so its points are
  in-sample (PR-AUC 0.0114 on a 0.129% fraud rate).
- **Run 2 onward** are evaluated on held-out rows only (PR-AUC 0.0280 on a
  0.339% fraud rate).

The jump in PR-AUC from run 1 to run 2 comes from the higher fraud rate in
the held-out window, not from a better model; ROC-AUC, which doesn't depend
on the fraud rate, is flat (0.854 in both). Compare PR-AUC only between
runs evaluated on the same window. See
[07-evaluation-metrics.md](07-evaluation-metrics.md#generalization-check-overfitting).
The view doesn't plot the training-window metrics (`train_pr_auc`,
`train_roc_auc`); query them from `model_runs` directly.

---
[← 08 — Drift Monitoring](08-drift-monitoring.md) | [Index](README.md) | Next: [10 — Testing & Validation →](10-testing-and-validation.md)
