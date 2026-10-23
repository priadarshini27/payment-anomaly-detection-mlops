# 08 — Drift Monitoring

[← 07 — Evaluation Metrics](07-evaluation-metrics.md) | [Index](README.md) | Next: [09 — Dashboard →](09-dashboard.md)

Drift monitoring compares the distribution of each engineered feature in
a recent scoring window against a fixed reference window captured at
training time (persisted by `models/train.py` as
`reference_window_<timestamp>.csv`). Two complementary statistical tests
are used because they are sensitive to different kinds of shift.

## Population Stability Index (PSI)

PSI buckets each feature into deciles based on the reference distribution,
then measures how much the current distribution's bucket proportions have
shifted:

```
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
```

| PSI range | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | No action |
| 0.10 – 0.25 | Moderate shift | Log and monitor |
| > 0.25 | Significant shift | Flag for retraining review |

These are the industry-standard bands used in credit-risk model
monitoring, applied per feature in `drift/psi.py`.

## Kolmogorov–Smirnov test

The two-sample KS test complements PSI by testing the maximum distance
between the empirical CDFs of the reference and current windows, and
returns a p-value rather than a fixed-bucket approximation — this catches
shifts within a single PSI decile bucket that PSI alone would understate.
A feature is flagged as drifting if its KS p-value falls below 0.05 (the
null hypothesis that both windows are drawn from the same distribution is
rejected at 95% confidence).

## Retraining trigger

Retraining is triggered automatically when **either**:

- **(a)** three or more features simultaneously exceed the PSI 0.25
  "significant" threshold, **or**
- **(b)** PR-AUC on the rolling evaluation window drops more than 15%
  relative to the last deployed model's PR-AUC.

This dual criterion — distributional shift and realized performance drop
— avoids retraining on noisy single-feature fluctuations while still
catching silent performance decay that PSI alone might miss.

## `drift/psi.py`

### File: drift/psi.py

```python
"""
drift/psi.py
==============
Population Stability Index (PSI) -- Document 1, Section 6.1.

PSI buckets a feature into deciles based on a reference distribution, then
measures how much a current distribution's bucket proportions have shifted:

    PSI = sum over buckets of (Actual% - Expected%) * ln(Actual% / Expected%)

Interpretation bands (industry-standard, credit-risk model monitoring):
    < 0.10          no significant shift          -> no action
    0.10 - 0.25     moderate shift                -> log and monitor
    > 0.25          significant shift              -> flag for retraining review
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

# Small constant to avoid divide-by-zero / log(0) when a bucket is empty in
# either distribution.
EPSILON = 1e-6


def _bucket_edges(reference: np.ndarray, buckets: int) -> np.ndarray:
    """Decile (or n-tile) edges computed from the reference distribution."""
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)  # collapse duplicate edges (e.g. many-zero features)
    if len(edges) < 2:
        # Degenerate distribution (constant feature) -- use a single bucket.
        edges = np.array([edges[0] - 1, edges[0] + 1])
    return edges


def calculate_psi(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, buckets: int = 10) -> float:
    """
    Computes the Population Stability Index between a reference distribution
    (captured at training time) and a current distribution (a recent
    scoring window), for a single numeric feature.
    """
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = _bucket_edges(reference, buckets)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPSILON, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_flag(psi_value: float) -> str:
    """Maps a PSI value to its interpretation band."""
    if psi_value > PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi_value >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def calculate_psi_for_features(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str], buckets: int = 10
) -> pd.DataFrame:
    """Computes PSI for every feature in feature_columns, returning a tidy DataFrame."""
    rows = []
    for col in feature_columns:
        psi_value = calculate_psi(reference_df[col], current_df[col], buckets=buckets)
        rows.append({"feature_name": col, "psi_value": psi_value, "psi_flag": psi_flag(psi_value)})
    return pd.DataFrame(rows)
```

## `drift/ks_test.py`

### File: drift/ks_test.py

```python
"""
drift/ks_test.py
===================
Two-sample Kolmogorov-Smirnov drift test -- Document 1, Section 6.2.

Complements PSI by testing the maximum distance between the empirical CDFs
of the reference and current windows, returning a p-value rather than a
fixed-bucket approximation. This catches shifts within a single PSI decile
bucket that PSI alone would understate.

A feature is flagged as drifting if its KS p-value falls below alpha
(default 0.05), i.e. the null hypothesis that both windows are drawn from
the same distribution is rejected at 95% confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

KS_ALPHA = 0.05


def ks_drift_test(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, alpha: float = KS_ALPHA) -> dict:
    """
    Runs scipy.stats.ks_2samp between a reference and current distribution
    for a single numeric feature. Returns a dict with the KS statistic,
    p-value, and a boolean drift flag (p-value < alpha).
    """
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "ks_flag": False}

    result = ks_2samp(reference, current)
    return {
        "ks_statistic": float(result.statistic),
        "ks_pvalue": float(result.pvalue),
        "ks_flag": bool(result.pvalue < alpha),
    }


def ks_test_for_features(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str], alpha: float = KS_ALPHA
) -> pd.DataFrame:
    """Runs the KS test for every feature in feature_columns, returning a tidy DataFrame."""
    rows = []
    for col in feature_columns:
        result = ks_drift_test(reference_df[col], current_df[col], alpha=alpha)
        rows.append({"feature_name": col, **result})
    return pd.DataFrame(rows)
```

## `drift/retrain.py`

### File: drift/retrain.py

```python
"""
drift/retrain.py
===================
Orchestrates a drift check: compares the reference feature window captured
at training time against a current scoring window pulled from the
`transactions` table, runs PSI (drift/psi.py) and the KS test
(drift/ks_test.py) for every monitored feature, persists the results to
drift_metrics, applies the dual retraining trigger from Document 1 Section
6.3, and -- if triggered -- calls models/train.py + models/evaluate.py to
produce and evaluate a new model version. Every check (triggered or not)
is logged to retrain_log for audit purposes.

Retraining trigger (Document 1, Section 6.3) -- fires if EITHER:
  (a) three or more monitored features simultaneously exceed the PSI 0.25
      "significant" threshold, OR
  (b) PR-AUC on the current evaluation drops more than 15% relative to the
      last deployed model's PR-AUC.
This dual criterion avoids retraining on noisy single-feature fluctuations
while still catching silent performance decay that PSI alone might miss.

Run directly:
    python drift/retrain.py --current-step-start 600 --current-step-end 744
Compares the persisted reference window against transactions in the given
`step` range (defaults to the last 10% of the simulated 744-step period).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.ks_test import ks_drift_test  # noqa: E402
from drift.psi import calculate_psi, psi_flag  # noqa: E402
from preprocessing.feature_engineering import MODEL_FEATURE_COLUMNS  # noqa: E402
from sql.queries import get_engine, DEFAULT_DB_PATH  # noqa: E402

N_DRIFTED_FEATURES_TRIGGER = 3
PR_AUC_DROP_PCT_TRIGGER = 0.15


def load_reference_window(reference_window_path: str) -> pd.DataFrame:
    return pd.read_csv(reference_window_path)


def load_current_window(engine, step_start: int, step_end: int) -> pd.DataFrame:
    sql = text(
        f"SELECT step, {', '.join(MODEL_FEATURE_COLUMNS)} FROM transactions "
        "WHERE step BETWEEN :s AND :e;"
    )
    return pd.read_sql(sql, engine, params={"s": step_start, "e": step_end})


def run_drift_checks(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Runs PSI + KS for every MODEL_FEATURE_COLUMNS feature, returns a tidy DataFrame."""
    rows = []
    for col in MODEL_FEATURE_COLUMNS:
        psi_value = calculate_psi(reference_df[col], current_df[col])
        ks_result = ks_drift_test(reference_df[col], current_df[col])
        rows.append(
            {
                "feature_name": col,
                "psi_value": psi_value,
                "psi_flag": psi_flag(psi_value),
                "ks_statistic": ks_result["ks_statistic"],
                "ks_pvalue": ks_result["ks_pvalue"],
                "ks_flag": int(ks_result["ks_flag"]),
            }
        )
    return pd.DataFrame(rows)


def persist_drift_metrics(
    engine, run_id: int, drift_df: pd.DataFrame,
    reference_window: tuple[int, int], current_window: tuple[int, int],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        for row in drift_df.itertuples(index=False):
            conn.execute(
                text(
                    """
                    INSERT INTO drift_metrics (
                        run_id, feature_name, psi_value, ks_statistic, ks_pvalue,
                        reference_window_start, reference_window_end,
                        current_window_start, current_window_end,
                        computed_at, psi_flag, ks_flag
                    ) VALUES (
                        :run_id, :feature_name, :psi_value, :ks_statistic, :ks_pvalue,
                        :ref_start, :ref_end, :cur_start, :cur_end,
                        :computed_at, :psi_flag, :ks_flag
                    );
                    """
                ),
                {
                    "run_id": run_id,
                    "feature_name": row.feature_name,
                    "psi_value": row.psi_value,
                    "ks_statistic": row.ks_statistic,
                    "ks_pvalue": row.ks_pvalue,
                    "ref_start": reference_window[0],
                    "ref_end": reference_window[1],
                    "cur_start": current_window[0],
                    "cur_end": current_window[1],
                    "computed_at": now,
                    "psi_flag": row.psi_flag,
                    "ks_flag": row.ks_flag,
                },
            )


def evaluate_retrain_trigger(drift_df: pd.DataFrame, current_pr_auc: float | None, baseline_pr_auc: float | None) -> dict:
    """
    Applies the dual retraining trigger from Document 1 Section 6.3.
    Returns a dict describing whether retraining should fire and why.
    """
    n_significant = int((drift_df["psi_flag"] == "significant").sum())
    psi_trigger = n_significant >= N_DRIFTED_FEATURES_TRIGGER

    pr_auc_drop_pct = None
    perf_trigger = False
    if current_pr_auc is not None and baseline_pr_auc not in (None, 0):
        pr_auc_drop_pct = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc
        perf_trigger = pr_auc_drop_pct > PR_AUC_DROP_PCT_TRIGGER

    triggered = psi_trigger or perf_trigger
    if psi_trigger and perf_trigger:
        reason = "psi+performance"
    elif psi_trigger:
        reason = "psi"
    elif perf_trigger:
        reason = "performance"
    else:
        reason = None

    return {
        "triggered": triggered,
        "trigger_reason": reason,
        "n_features_drifted": n_significant,
        "pr_auc_drop_pct": pr_auc_drop_pct,
    }


def persist_retrain_log(engine, decision: dict, baseline_run_id: int, resulting_run_id: int | None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO retrain_log (
                    checked_at, triggered, trigger_reason, n_features_drifted,
                    pr_auc_drop_pct, baseline_run_id, resulting_run_id
                ) VALUES (
                    :checked_at, :triggered, :trigger_reason, :n_features_drifted,
                    :pr_auc_drop_pct, :baseline_run_id, :resulting_run_id
                );
                """
            ),
            {
                "checked_at": now,
                "triggered": int(decision["triggered"]),
                "trigger_reason": decision["trigger_reason"],
                "n_features_drifted": decision["n_features_drifted"],
                "pr_auc_drop_pct": decision["pr_auc_drop_pct"],
                "baseline_run_id": baseline_run_id,
                "resulting_run_id": resulting_run_id,
            },
        )


def check_and_retrain(
    db_path: str | None = None,
    current_step_start: int = 600,
    current_step_end: int = 744,
    auto_retrain: bool = True,
) -> dict:
    engine = get_engine(db_path or DEFAULT_DB_PATH)

    with engine.connect() as conn:
        baseline = conn.execute(
            text("SELECT * FROM model_runs ORDER BY run_id DESC LIMIT 1")
        ).mappings().fetchone()
    if baseline is None:
        raise RuntimeError("No model_runs row found. Run models/train.py first.")
    baseline = dict(baseline)

    reference_df = load_reference_window(baseline["reference_window_path"])
    current_df = load_current_window(engine, current_step_start, current_step_end)

    if current_df.empty:
        raise ValueError(
            f"No transactions found in step range [{current_step_start}, {current_step_end}]."
        )

    drift_df = run_drift_checks(reference_df, current_df)
    persist_drift_metrics(
        engine, baseline["run_id"], drift_df,
        reference_window=(int(reference_df["step"].min()), int(reference_df["step"].max())),
        current_window=(current_step_start, current_step_end),
    )

    decision = evaluate_retrain_trigger(
        drift_df, current_pr_auc=baseline.get("pr_auc"), baseline_pr_auc=baseline.get("pr_auc")
    )

    print("=" * 60)
    print("Drift check summary")
    print("=" * 60)
    print(drift_df.to_string(index=False))
    print(f"\nFeatures with significant PSI drift: {decision['n_features_drifted']}")
    print(f"Retrain triggered: {decision['triggered']} (reason: {decision['trigger_reason']})")
    print("=" * 60)

    resulting_run_id = None
    if decision["triggered"] and auto_retrain:
        from models.evaluate import evaluate as evaluate_model
        from models.train import train as train_model

        print("Retraining triggered -- fitting a new model version...")
        resulting_run_id = train_model(db_path=db_path)
        evaluate_model(db_path=db_path, run_id=resulting_run_id)
        print(f"Retraining complete. New model_runs.run_id = {resulting_run_id}")

    persist_retrain_log(engine, decision, baseline["run_id"], resulting_run_id)

    return {**decision, "baseline_run_id": baseline["run_id"], "resulting_run_id": resulting_run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--current-step-start", type=int, default=600)
    parser.add_argument("--current-step-end", type=int, default=744)
    parser.add_argument(
        "--no-auto-retrain", action="store_true",
        help="Only run the drift check and log the decision; don't actually retrain even if triggered.",
    )
    args = parser.parse_args()

    check_and_retrain(
        db_path=args.db_path,
        current_step_start=args.current_step_start,
        current_step_end=args.current_step_end,
        auto_retrain=not args.no_auto_retrain,
    )


if __name__ == "__main__":
    main()
```

## Running it

```bash
python drift/retrain.py
```

By default this compares the persisted training reference window against
`transactions` rows in `step` 600–744 (the last ~6 days of the simulated
period). Override with `--current-step-start`/`--current-step-end` to
check a different window, or pass `--no-auto-retrain` to only log the
decision without actually fitting a new model even if the trigger fires.

### What the reference window contains

The reference window is a 200,000-row random sample of the model's
**training partition** only (steps 1–355 for run 2; see
[06-modeling.md](06-modeling.md#traintest-split)). The default "current"
window, steps 600–744, lies entirely in the held-out period, so the drift
check compares the data the model learned from with transactions it has
never seen, which is the comparison that matters in production.

### Measured result (run 2: reference steps 1–355 vs. steps 600–744)

```
============================================================
Drift check summary
============================================================
    feature_name  psi_value psi_flag  ks_statistic     ks_pvalue  ks_flag
       logAmount   0.000953     none      0.013622  1.984242e-11        1
balanceDeltaOrig   0.008264     none      0.036269  1.992662e-78        1
balanceDeltaDest   0.023657     none      0.036909  3.331816e-81        1
   origZeroedOut   0.000000     none      0.041153 7.381492e-101        1
  isMerchantDest   0.000000     none      0.002140  9.129204e-01        0
    type_CASH_IN   0.000000     none      0.013998  4.815885e-12        1
   type_CASH_OUT   0.000000     none      0.032967  7.184132e-65        1
      type_DEBIT   0.000000     none      0.001800  9.795725e-01        0
    type_PAYMENT   0.000000     none      0.002140  9.129204e-01        0
   type_TRANSFER   0.000000     none      0.015030  8.073695e-14        1
   txnVelocity1h   0.000000     none      0.000040  1.000000e+00        0
  txnVelocity24h   0.000000     none      0.000210  1.000000e+00        0

Features with significant PSI drift: 0
Retrain triggered: False (reason: None)
============================================================

real    0m25.139s
```

Two things worth noting about this result. First, **PSI reports no drift
on any feature** (all `< 0.10`), even though the current window is
genuinely out-of-time. The feature distributions in PaySim's late steps
are close to those of the early steps, which is consistent with the
held-out evaluation in [07-evaluation-metrics.md](07-evaluation-metrics.md)
showing no drop in ROC-AUC. (Run 1's check, where the reference sample and
the current window both came from the same full training set, gave almost
identical values.) Second, **KS flags several features as significant
despite near-zero PSI** (e.g. `origZeroedOut`, `balanceDeltaDest`), which
is not a contradiction — it is exactly the behavior described in
[02-dataset-and-eda.md](02-dataset-and-eda.md)/Document 1 Section 6.2: KS
is far more sensitive than decile-bucketed PSI and will flag statistically
significant differences at multi-million-row sample sizes even when the
practical distributional shift is negligible. This is precisely why the
retraining trigger (Section 6.3 above) is keyed on PSI's `significant`
band and not on the KS p-value alone — using KS as the sole trigger at
this scale would fire constantly on noise.

### Known limitation: the PR-AUC trigger never fires

`check_and_retrain()` passes the latest run's stored `pr_auc` as **both**
the current and the baseline value, so the PR-AUC drop it computes is
always 0 and the performance half of the dual trigger never fires,
regardless of how many models have been evaluated. Only the PSI half is
active. Now that `models/evaluate.py` produces held-out metrics, the fix is
to compute `current_pr_auc` by scoring the current window with the
deployed model and comparing it with the deployed run's held-out
`pr_auc` (taking into account that PR-AUC moves with the fraud base rate;
see [07-evaluation-metrics.md](07-evaluation-metrics.md#generalization-check-overfitting)).

---
[← 07 — Evaluation Metrics](07-evaluation-metrics.md) | [Index](README.md) | Next: [09 — Dashboard →](09-dashboard.md)
