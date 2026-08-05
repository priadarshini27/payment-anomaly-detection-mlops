# 11 — Deployment & Operations

[← 10 — Testing & Validation](10-testing-and-validation.md) | [Index](README.md) | Next: [12 — Troubleshooting →](12-troubleshooting.md)

This project is designed to run as a set of independently schedulable
batch jobs plus a long-running dashboard process — there is no always-on
API server to deploy.

## Recommended operational cadence

| Job | Command | Suggested cadence |
|---|---|---|
| Ingest + clean new data | `python preprocessing/clean_data.py` | Whenever new source data lands (e.g. nightly) |
| Backfill velocity features | `python sql/queries.py` | Immediately after any ingest that adds rows |
| Score new transactions | `python models/predict.py` | After every ingest, or continuously in a streaming variant (out of scope here — see Further Study below) |
| Drift check + conditional retrain | `python drift/retrain.py --current-step-start <N> --current-step-end <M>` | Daily, or after every ingest batch, comparing the most recent window against the standing reference |
| Dashboard | `streamlit run dashboard/app.py` | Long-running, always on |

### Example: a simple daily cron-style driver script

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /path/to/payment-anomaly-detection-mlops
source .venv/bin/activate

python preprocessing/clean_data.py
python sql/queries.py
python drift/retrain.py --current-step-start "$(( $(date +%s) / 3600 % 744 - 24 ))" \
                         --current-step-end "$(( $(date +%s) / 3600 % 744 ))"
```

In a real deployment, replace the ad-hoc `step`-range arithmetic with
whatever maps your live transaction timestamps to PaySim-style hourly
`step` values, or — if you're operating on genuinely live data rather
than the static PaySim simulation — replace the `step`-based windowing in
`drift/retrain.py::load_current_window` with a real timestamp column.

## Configuration

All configuration in this project is expressed as function parameters and
CLI flags rather than environment variables or a config file, since every
script is small enough that this stays readable. The knobs that matter
operationally:

| Parameter | Where | Default | Purpose |
|---|---|---|---|
| `--db-path` | every CLI script | `database/database.db` | Point at a different SQLite file (e.g. for a staging environment) |
| `--chunksize` | `clean_data.py`, `load_data.py` | 500,000 / 100,000 | Lower this on memory-constrained machines |
| `HYPERPARAMS` dict | `models/train.py` | see [06-modeling.md](06-modeling.md) | Model hyperparameters — edit in code and re-run to change |
| `K_FRACTION` | `models/evaluate.py` | 0.005 | The "top k%" definition for precision@k/recall@k |
| `N_DRIFTED_FEATURES_TRIGGER`, `PR_AUC_DROP_PCT_TRIGGER` | `drift/retrain.py` | 3, 0.15 | The dual retraining trigger thresholds |
| `--current-step-start`/`--current-step-end` | `drift/retrain.py` | 600 / 744 | The scoring window compared against the training reference |

## Production hardening notes

- **SQLite concurrency.** SQLite handles concurrent readers well but
  serializes writers. The dashboard only reads; batch jobs (`clean_data.py`,
  `load_data.py`, `train.py`, `predict.py`, `retrain.py`) write. Avoid
  running two write jobs simultaneously against the same `database.db` —
  if a real multi-writer deployment is needed, migrate to PostgreSQL
  (the SQLAlchemy layer in `sql/queries.py` makes this a connection-string
  change, not a query rewrite, since all queries use standard SQL).
- **Model artifact retention.** `models/train.py` never deletes old
  artifacts under `models/artifacts/` — every version is kept. In a long
  -running deployment, add a retention policy (e.g. keep the last N
  versions, or only versions referenced by a `model_runs` row less than
  90 days old) to bound disk usage.
- **Backing up `database.db`.** Since it is a single file, backups are a
  file copy:
  ```bash
  cp database/database.db "backups/database_$(date +%Y%m%dT%H%M%S).db"
  ```
  Do this before any schema change or before a bulk reload
  (`database/load_data.py` without `--no-reset` truncates `transactions`).
- **Secrets.** This project has none — no API keys, no credentials. If a
  future iteration adds a real data source (an API, a message queue),
  keep credentials out of the scripts and in environment variables loaded
  via `os.environ`, never hardcoded into `sql/queries.py` or elsewhere.

## Further study

These are natural next steps beyond what is implemented here, listed so
a future contributor knows where the deliberate scope boundary is:

| Concept | Resource |
|---|---|
| Feature Engineering | https://www.kaggle.com/learn/feature-engineering |
| Isolation Forest | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html |
| Model Evaluation Metrics | https://scikit-learn.org/stable/modules/model_evaluation.html |
| Drift Detection (KS Test) | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html |
| Streamlit Dashboard | https://docs.streamlit.io |
| Plotly Charts | https://plotly.com/python/ |

---
[← 10 — Testing & Validation](10-testing-and-validation.md) | [Index](README.md) | Next: [12 — Troubleshooting →](12-troubleshooting.md)
