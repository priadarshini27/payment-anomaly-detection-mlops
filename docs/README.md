# Payment Anomaly Detection MLOps — Documentation Library

This is the single source of truth for the **Payment Transaction Anomaly
Detection with Data Drift Monitoring** project. It is written so that a
developer with **no access to the original repository** can recreate the
entire project — every folder, every file, every line of source code,
every database table, every command — by following these documents in
order and copy-pasting the code blocks they contain.

The original project specification lives in two Word documents under
`doc/` (`Doc1_Project_Overview.docx`, `Doc2_Exploratory_Data_Analysis.docx`).
This library is the executable, code-complete realization of that
specification, cross-referenced back to it throughout.

A one-document summary of everything below — [Project_Overview.docx](Project_Overview.docx)
— covers the same as-built architecture, measured results, and performance
lessons in Word format with the system architecture diagram embedded,
generated from this library via `_build_project_overview_docx.py`
(requires `pip install python-docx`; not a runtime dependency).

## How to use this library

- If you are setting the project up for the first time, start at
  [01-getting-started.md](01-getting-started.md) and follow the documents
  in numeric order — each one builds on artifacts produced by the
  previous one (the cleaned CSV feeds the database, the database feeds
  the model, the model feeds drift monitoring, everything feeds the
  dashboard).
- If you already have the project running and need to understand or
  modify one specific layer, jump directly to the relevant document using
  the index below.
- Every document that introduces a source file states its **exact
  repo-relative path** as a heading immediately above the code block, and
  states the exact command used to run it. Nothing is left implied.

## Document Index

| # | Document | Covers |
|---|----------|--------|
| 00 | [00-overview.md](00-overview.md) | Project purpose, architecture diagram, tech stack, module map |
| 01 | [01-getting-started.md](01-getting-started.md) | Prerequisites, environment setup, folder scaffolding, first run |
| 02 | [02-dataset-and-eda.md](02-dataset-and-eda.md) | PaySim schema, class imbalance, EDA methodology and findings |
| 03 | [03-data-cleaning.md](03-data-cleaning.md) | Cleaning rules, feature engineering, full source of `preprocessing/` |
| 04 | [04-database.md](04-database.md) | Full SQLite schema, table reference, full source of `database/` |
| 05 | [05-sql-analytics.md](05-sql-analytics.md) | Full source of `sql/queries.py`, query-by-query explanation |
| 06 | [06-modeling.md](06-modeling.md) | Isolation Forest configuration, full source of `models/` |
| 07 | [07-evaluation-metrics.md](07-evaluation-metrics.md) | Why accuracy is excluded; PR-AUC, ROC-AUC, precision@k, F1 |
| 08 | [08-drift-monitoring.md](08-drift-monitoring.md) | PSI, KS test, retrain trigger, full source of `drift/` |
| 09 | [09-dashboard.md](09-dashboard.md) | All 6 dashboard views, full source of `dashboard/app.py` |
| 10 | [10-testing-and-validation.md](10-testing-and-validation.md) | Full test source, how to run `pytest`, manual validation checklist |
| 11 | [11-deployment-and-operations.md](11-deployment-and-operations.md) | Scheduling retraining, config, production hardening, backups |
| 12 | [12-troubleshooting.md](12-troubleshooting.md) | Common errors and fixes |

## Conventions used throughout this library

- **File headers.** Every code block is preceded by a line of the form
  `### File: path/to/file.py` giving the exact repo-relative path to
  create. Create the file at that path and paste the block verbatim.
- **Commands.** Shell commands are given for both PowerShell (Windows,
  this project's primary development environment) and POSIX shells
  (macOS/Linux/Git Bash), where they differ.
- **Full scale.** All pipeline runs described in this library were
  executed against the **full PaySim dataset** (6,362,620 raw rows), not
  a sample. Row counts and metrics quoted in these documents are the
  real, measured output of those runs — not illustrative placeholders.
