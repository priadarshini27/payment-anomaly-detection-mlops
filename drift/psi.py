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
