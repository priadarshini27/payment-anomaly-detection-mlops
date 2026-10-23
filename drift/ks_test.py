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
