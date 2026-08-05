"""
docs/_build_project_overview_docx.py
=======================================
One-off generator for docs/Project_Overview.docx. Not part of the
pipeline -- run manually (`python docs/_build_project_overview_docx.py`)
whenever the overview needs to be regenerated from the docs/ markdown
library after a significant change. Requires python-docx
(`pip install python-docx`; not in requirements.txt since it's a docs
authoring tool, not a runtime dependency).

Source of truth for the numbers below is the actual pipeline run recorded
in docs/05 through docs/08 and verified against database/database.db --
this document describes what was built and measured, not the original
Doc1_Project_Overview.docx planning spec (which used illustrative/target
figures and a slightly different module layout than what was ultimately
implemented, e.g. preprocessing/clean_data.py rather than
preprocessing/ingest.py, and sql/queries.py as a single module rather than
sql/queries/*.sql).
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

DOCS_DIR = Path(__file__).resolve().parent
DIAGRAM_PATH = DOCS_DIR / "architecture-diagram.png"
OUTPUT_PATH = DOCS_DIR / "Project_Overview.docx"

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
GRAY = RGBColor(0x55, 0x55, 0x55)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY
    return h


def add_body(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(item, tuple):
            label, rest = item
            r1 = p.add_run(label)
            r1.bold = True
            p.add_run(rest)
        else:
            p.add_run(item)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return table


def build():
    doc = Document()

    # ---- Title page -------------------------------------------------
    title = doc.add_heading("Payment Anomaly Detection MLOps", level=0)
    for run in title.runs:
        run.font.color.rgb = NAVY
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Project Overview — As-Built")
    run.font.size = Pt(16)
    run.font.color.rgb = GRAY
    sub2 = doc.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = sub2.add_run(
        "Unsupervised fraud detection on PaySim, with SQL analytics, drift monitoring, "
        "and a Streamlit dashboard — documenting the pipeline as implemented and run "
        "against the full 6,362,604-row dataset."
    )
    run2.italic = True
    run2.font.color.rgb = GRAY
    doc.add_page_break()

    # ---- 1. Summary ---------------------------------------------------
    add_heading(doc, "1. What This Is", level=1)
    add_body(
        doc,
        "A production-style machine learning system that detects anomalous payment "
        "transactions in the PaySim mobile-money dataset using an unsupervised "
        "Isolation Forest, persists and queries the data through a normalized SQLite "
        "schema, monitors feature-level drift with PSI and the Kolmogorov-Smirnov "
        "test, and surfaces everything through an interactive Streamlit dashboard. "
        "The full pipeline -- ingestion through dashboard -- has been run end-to-end "
        "against the complete PaySim release (6,362,604 transactions after cleaning), "
        "not a sample."
    )
    add_bullets(doc, [
        ("Detects fraud without labels: ", "the model trains only on engineered transaction "
         "features; isFraud is used exclusively for offline evaluation, never as a model input."),
        ("Full-scale, not a demo: ", "every number in this document (row counts, timing, PR-AUC) "
         "is measured from an actual run against the full dataset, not a subsample."),
        ("Self-monitoring: ", "drift/retrain.py compares live scoring windows against the "
         "training reference distribution and can trigger retraining automatically."),
        ("Documented for independent rebuild: ", "docs/00 through docs/12 contain every file's "
         "complete source, so the project can be recreated without this repository."),
    ])

    # ---- 2. Architecture diagram --------------------------------------
    add_heading(doc, "2. System Architecture", level=1)
    add_body(
        doc,
        "The diagram below is the original design-time architecture (from the project's "
        "planning document) and remains accurate at the level of data flow: ingestion into "
        "SQLite, SQL analytics and feature engineering, Isolation Forest scoring, evaluation, "
        "drift monitoring feeding back into retraining, and a Streamlit dashboard consuming "
        "all of it. Section 4 below notes where the as-built module layout differs in detail "
        "from this diagram's labels."
    )
    if DIAGRAM_PATH.exists():
        doc.add_picture(str(DIAGRAM_PATH), width=Inches(6.5))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run("Figure 1. End-to-end system architecture and data flow.")
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = GRAY
    doc.add_paragraph()

    # ---- 3. Dataset -----------------------------------------------------
    add_heading(doc, "3. Dataset", level=1)
    add_body(
        doc,
        "PaySim is a synthetic mobile-money transaction log generated from an agent-based "
        "simulation calibrated against a real African mobile financial service provider's "
        "transaction data. It spans a simulated 30-day period (744 hourly steps) across five "
        "transaction types and is a standard fraud-detection benchmark because of its extreme "
        "class imbalance and internally consistent account-balance structure."
    )
    add_table(
        doc,
        ["Field", "Type", "Description"],
        [
            ("step", "int", "Simulation time unit; 1 step = 1 hour (1-744)."),
            ("type", "categorical", "CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER."),
            ("amount", "float", "Transaction value in local currency."),
            ("nameOrig", "string", "Originating customer identifier."),
            ("oldbalanceOrg / newbalanceOrig", "float", "Originator balance before / after."),
            ("nameDest", "string", "Recipient identifier."),
            ("oldbalanceDest / newbalanceDest", "float", "Recipient balance before / after (0 for merchants)."),
            ("isFraud", "binary", "Ground-truth label -- evaluation only, never a model feature."),
            ("isFlaggedFraud", "binary", "PaySim's own rule-based flag."),
        ],
    )
    add_body(doc, "As measured on this project's cleaned, full-scale dataset:", bold=True)
    add_bullets(doc, [
        ("6,362,604 rows", " after cleaning (16 zero-amount fraudulent rows dropped from the "
         "raw 6,362,620; zero missing values, zero duplicates in the source)."),
        ("8,197 fraudulent transactions", " -- a base rate of 0.129%, confined almost entirely "
         "to TRANSFER and CASH_OUT."),
        ("6,353,291 distinct originators", " across those 6.36M rows (at most 3 transactions "
         "per originator) -- a data characteristic that directly shaped the velocity-feature "
         "implementation; see Section 5."),
    ])

    # ---- 4. As-built module layout --------------------------------------
    add_heading(doc, "4. As-Built Module Layout", level=1)
    add_body(
        doc,
        "The table below is the actual repository structure as implemented and run, which "
        "differs in a few specifics from the original planning document: SQL analytics live "
        "in a single sql/queries.py module (not per-query .sql files), ingestion is "
        "preprocessing/clean_data.py + database/load_data.py rather than a single "
        "preprocessing/ingest.py, and the model module is models/train.py rather than "
        "models/isolation_forest.py. Every module below has a corresponding docs/0N file with "
        "its complete source."
    )
    add_table(
        doc,
        ["Layer", "Module(s)", "Responsibility"],
        [
            ("Ingestion", "preprocessing/clean_data.py,\ndatabase/load_data.py",
             "Chunked CSV read, cleaning, schema validation, bulk load into SQLite."),
            ("Storage", "database/schema.sql",
             "Normalized tables: transactions, model_runs, drift_metrics, retrain_log."),
            ("Analytics", "sql/queries.py",
             "Parameterized SQL + vectorized velocity-feature computation, dashboard queries."),
            ("Feature Eng.", "preprocessing/feature_engineering.py",
             "Balance-delta residuals, log-amount, one-hot type, velocity features."),
            ("Modeling", "models/train.py",
             "Fits sklearn IsolationForest, persists model + scaler via joblib."),
            ("Scoring", "models/predict.py",
             "Scores transactions, bulk-writes anomaly_score/is_anomaly back to SQLite."),
            ("Evaluation", "models/evaluate.py",
             "PR-AUC, ROC-AUC, precision@k/recall@k, confusion matrix by type."),
            ("Drift", "drift/psi.py, drift/ks_test.py,\ndrift/retrain.py",
             "PSI + KS per feature against a fixed reference window; retraining trigger."),
            ("Dashboard", "dashboard/app.py",
             "Streamlit app, six views, backed by live SQLite queries."),
            ("Tests", "tests/", "24 pytest cases covering cleaning, feature engineering, drift."),
        ],
    )

    # ---- 5. Feature engineering & modeling ------------------------------
    add_heading(doc, "5. Feature Engineering and Modeling", level=1)
    add_body(doc, "Twelve engineered features feed the model (MODEL_FEATURE_COLUMNS):", bold=True)
    add_bullets(doc, [
        ("balanceDeltaOrig / balanceDeltaDest: ", "non-zero residual between the expected and "
         "actual post-transaction balance -- a strong fraud signal PaySim's generator "
         "introduces deliberately."),
        ("origZeroedOut: ", "binary flag for a full-balance drain, typical of account-takeover fraud."),
        ("logAmount: ", "log1p(amount), stabilizing the heavily right-skewed amount distribution."),
        ("type_CASH_IN / _CASH_OUT / _DEBIT / _PAYMENT / _TRANSFER: ", "one-hot transaction type."),
        ("isMerchantDest: ", "flag for merchant-prefixed recipient identifiers."),
        ("txnVelocity1h / txnVelocity24h: ", "count of an originator's transactions within a "
         "1-hour / 24-hour trailing window."),
    ])
    add_body(
        doc,
        "isFraud and isFlaggedFraud are never included in the feature set -- that is what "
        "makes this an unsupervised detector rather than a supervised classifier in disguise."
    )
    add_body(doc, "Isolation Forest hyperparameters:", bold=True)
    add_table(
        doc,
        ["Hyperparameter", "Value", "Rationale"],
        [
            ("n_estimators", "200", "Stabilizes path-length averages beyond sklearn's default of 100."),
            ("contamination", "0.0013", "Set to the empirical training-partition fraud rate, not sklearn's 'auto'."),
            ("max_samples", "'auto' (256)", "Keeps isolation paths short and training tractable at multi-million-row scale."),
            ("max_features", "1.0", "All 12 features used per split; the set is small enough that sub-sampling is unnecessary."),
            ("random_state", "42", "Fixed for reproducibility across train/eval/retrain runs."),
        ],
    )

    # ---- 6. Measured results ---------------------------------------------
    add_heading(doc, "6. Measured Results (Full 6,362,604-Row Run)", level=1)
    add_body(
        doc,
        "Every figure below is from an actual run against the complete dataset, recorded in "
        "docs/05 through docs/08 alongside full command output."
    )
    add_table(
        doc,
        ["Stage", "Result", "Time"],
        [
            ("Velocity feature backfill", "6,362,604 rows updated", "4m 25s"),
            ("Model training", "run_id = 1, 200 estimators", "2m 58s"),
            ("Evaluation", "PR-AUC 0.0114, ROC-AUC 0.8540", "6m 9s"),
            ("Anomaly scoring", "6,362,604 rows scored, 8,249 flagged", "7m 23s"),
            ("Drift check", "12 features checked, 0 significant PSI drift", "8.6s"),
        ],
    )
    add_body(
        doc,
        "A PR-AUC of 0.0114 against a 0.129% base fraud rate represents roughly an 8.8x "
        "enrichment over random guessing in the model's highest-scored transactions -- the "
        "absolute value looks small in isolation but is the expected order of magnitude under "
        "this severity of class imbalance. ROC-AUC of 0.854 confirms solid overall rank-"
        "ordering. The confusion matrix by type concentrates almost all errors in TRANSFER "
        "(26,793 false positives, 814 true positives) and CASH_OUT (291 false positives, 3 "
        "true positives) -- exactly the two types where PaySim fraud actually occurs, rather "
        "than spreading across CASH_IN / DEBIT / PAYMENT, which would have signaled a "
        "mis-specified feature."
    )
    add_body(
        doc,
        "Note: this evaluation run scored the model against the same full dataset it was "
        "trained on (no held-out nameOrig or step-range split), consistent with using isFraud "
        "only for offline evaluation rather than for train/test partitioning of an unsupervised "
        "model. These numbers describe in-sample discriminative power; a rigorous production "
        "benchmark would score a genuinely unseen window instead.",
        italic=True,
        size=10,
    )

    # ---- 7. Two performance lessons ---------------------------------------
    add_heading(doc, "7. Two Performance Lessons From Building This at Full Scale", level=1)
    add_body(
        doc,
        "Two implementation approaches that looked reasonable at design time failed to "
        "complete against the real 6.36M-row dataset, and fixing them shaped the final code "
        "in sql/queries.py, models/train.py, models/predict.py, and models/evaluate.py."
    )
    add_body(doc, "Lesson 1 -- the write pattern dominates, not the computation:", bold=True)
    add_body(
        doc,
        "Computing txnVelocity1h/24h as two correlated SQL subqueries per row did not finish "
        "in over 20 minutes, even though each subquery only touched 1-3 rows (nameOrig is "
        "nearly 1:1 with row count in this dataset). Replacing the subqueries with a fast "
        "vectorized numpy computation but still writing results back with one UPDATE statement "
        "per row also did not finish in reasonable time -- 6.36 million individual "
        "parameterized statements is itself the bottleneck, independent of how the values were "
        "computed. The fix: bulk-INSERT the computed values into a temp table, then apply all "
        "6.36 million values with a single set-based UPDATE...FROM-style join. Result: 4m 25s."
    )
    add_body(doc, "Lesson 2 -- pandas.read_sql builds an intermediate object array first:", bold=True)
    add_body(
        doc,
        "Loading the full feature matrix with pandas.read_sql(\"SELECT * FROM transactions\") "
        "raised numpy._core._exceptions.ArrayMemoryError: Unable to allocate 728 MiB for an "
        "array with shape (6362604, 15) and data type object. read_sql builds one large "
        "intermediate array of Python object tuples (one per row) before converting to typed "
        "columns; at this scale that intermediate representation alone needed hundreds of "
        "megabytes on top of the final array. The fix: stream through the raw DB-API cursor in "
        "fixed-size batches, writing each batch directly into a pre-allocated typed float64 "
        "array. Applied to models/train.py, models/predict.py, and models/evaluate.py."
    )

    # ---- 8. Drift monitoring -----------------------------------------------
    add_heading(doc, "8. Drift Monitoring", level=1)
    add_body(
        doc,
        "Drift monitoring compares each engineered feature's distribution in a recent scoring "
        "window against the fixed reference window captured at training time, using two "
        "complementary statistics."
    )
    add_table(
        doc,
        ["PSI Range", "Interpretation", "Action"],
        [
            ("< 0.10", "No significant shift", "No action"),
            ("0.10 - 0.25", "Moderate shift", "Log and monitor"),
            ("> 0.25", "Significant shift", "Flag for retraining review"),
        ],
    )
    add_body(
        doc,
        "The Kolmogorov-Smirnov test complements PSI by testing the maximum distance between "
        "empirical CDFs and returning a p-value rather than a fixed-bucket approximation. "
        "Retraining triggers automatically when three or more features simultaneously exceed "
        "PSI 0.25, or PR-AUC on the rolling evaluation window drops more than 15% relative to "
        "the last deployed model. On the measured run (reference window vs. step 600-744), all "
        "12 features showed zero significant PSI drift, so no retrain fired -- expected, since "
        "both windows are drawn from the same underlying training-set population. KS flagged "
        "several features as statistically significant despite near-zero PSI, which is not a "
        "contradiction: KS is far more sensitive at multi-million-row sample sizes and will "
        "flag negligible practical shifts as significant, which is exactly why the retraining "
        "trigger is keyed on PSI's banded severity rather than the KS p-value alone."
    )

    # ---- 9. Dashboard --------------------------------------------------------
    add_heading(doc, "9. Dashboard", level=1)
    add_table(
        doc,
        ["View", "Contents"],
        [
            ("Transaction Overview", "Volume and value trends by step and type; live SQL aggregates."),
            ("Fraud Analytics", "Flagged-transaction breakdown by type, amount band, and time."),
            ("Transaction Analysis", "Drill-down into individual transaction feature values and anomaly score."),
            ("Model Performance", "PR-AUC / ROC-AUC trend across model_runs versions; confusion matrix."),
            ("Drift Monitoring", "Per-feature PSI and KS p-value, with threshold bands overlaid."),
            ("System Health", "Pipeline run status, row counts, last retrain trigger reason."),
        ],
    )
    add_body(
        doc,
        "Verified: dashboard/app.py imports cleanly, every sql/queries.py function used by the "
        "dashboard returns correct results against the live 6.36M-row database, and a headless "
        "Streamlit launch (streamlit run dashboard/app.py --server.headless true) responds "
        "HTTP 200."
    )

    # ---- 10. Tech stack --------------------------------------------------
    add_heading(doc, "10. Tech Stack", level=1)
    add_table(
        doc,
        ["Category", "Technology"],
        [
            ("Language", "Python 3.11+"),
            ("Dataset", "PaySim"),
            ("Database", "SQLite"),
            ("Query Language", "SQL"),
            ("Data Processing", "Pandas, NumPy"),
            ("Machine Learning", "Scikit-Learn (IsolationForest)"),
            ("Statistics", "SciPy (ks_2samp)"),
            ("Drift Detection", "PSI (custom), KS Test (SciPy)"),
            ("Dashboard / Visualization", "Streamlit, Plotly"),
            ("Model Persistence", "joblib"),
            ("Testing", "pytest (24 tests)"),
        ],
    )

    # ---- 11. Verification summary ----------------------------------------
    add_heading(doc, "11. End-to-End Verification Summary", level=1)
    add_bullets(doc, [
        "6,362,604 / 6,362,604 rows populated with txnVelocity1h/24h -- zero NULLs.",
        "6,362,604 / 6,362,604 rows populated with anomaly_score/is_anomaly -- zero NULLs.",
        "model_runs: 1 row, is_deployed = 1, PR-AUC/ROC-AUC/precision@k/recall@k all populated.",
        "drift_metrics: 12 rows (one per model feature) for the evaluated run.",
        "retrain_log: 1 row, correctly logging no trigger on the reference-window comparison.",
        "pytest: 24/24 passing.",
        "Dashboard: imports cleanly, all query functions verified, headless launch returns HTTP 200.",
    ])
    add_body(
        doc,
        "Full command output, exact source for every module, and step-by-step rebuild "
        "instructions are in docs/00-overview.md through docs/12-troubleshooting.md.",
        italic=True,
    )

    doc.save(str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
