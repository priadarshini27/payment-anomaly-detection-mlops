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
