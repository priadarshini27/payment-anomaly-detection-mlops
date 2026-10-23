"""Unit tests for drift/psi.py and drift/ks_test.py."""

import numpy as np
import pandas as pd
import pytest

from drift.ks_test import ks_drift_test
from drift.psi import calculate_psi, psi_flag


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=5000)
    cur = ref.copy()
    psi = calculate_psi(ref, cur, buckets=10)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0, scale=1, size=5000)
    cur = rng.normal(loc=5, scale=1, size=5000)  # heavily shifted
    psi = calculate_psi(ref, cur, buckets=10)
    assert psi > 0.25
    assert psi_flag(psi) == "significant"


def test_psi_flag_bands():
    assert psi_flag(0.05) == "none"
    assert psi_flag(0.15) == "moderate"
    assert psi_flag(0.30) == "significant"


def test_psi_handles_empty_input_gracefully():
    assert calculate_psi(np.array([]), np.array([1, 2, 3])) == 0.0
    assert calculate_psi(np.array([1, 2, 3]), np.array([])) == 0.0


def test_ks_test_no_drift_for_identical_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    cur = ref.copy()
    result = ks_drift_test(ref, cur)
    assert result["ks_pvalue"] > 0.05
    assert result["ks_flag"] is False


def test_ks_test_detects_drift_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0, size=1000)
    cur = rng.normal(loc=3, size=1000)
    result = ks_drift_test(ref, cur)
    assert result["ks_pvalue"] < 0.05
    assert result["ks_flag"] is True


def test_ks_test_handles_tiny_samples_gracefully():
    result = ks_drift_test(np.array([1.0]), np.array([2.0]))
    assert result["ks_flag"] is False
