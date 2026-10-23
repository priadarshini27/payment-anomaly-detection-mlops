"""Unit tests for the time-based train/test split and generalization check."""

import numpy as np
import pandas as pd
import pytest

from models.evaluate import generalization_gap
from models.train import train_step_end_for_fraction


def make_step_counts(counts):
    return pd.DataFrame({"step": range(1, len(counts) + 1), "n": counts})


def test_split_cuts_at_requested_row_fraction():
    # 10 equal steps -> 80% of rows ends at step 8.
    assert train_step_end_for_fraction(make_step_counts([100] * 10), 0.8) == 8


def test_split_follows_rows_not_steps():
    # Early steps hold most rows, so 80% of rows is reached by step 2.
    assert train_step_end_for_fraction(make_step_counts([500, 400, 50, 50]), 0.8) == 2


def test_split_always_holds_out_at_least_one_step():
    assert train_step_end_for_fraction(make_step_counts([1000, 1]), 0.99) == 1


def test_split_is_order_independent():
    shuffled = make_step_counts([100] * 10).sample(frac=1, random_state=0)
    assert train_step_end_for_fraction(shuffled, 0.8) == 8


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        train_step_end_for_fraction(make_step_counts([100] * 10), fraction)


def test_split_rejects_single_step():
    with pytest.raises(ValueError):
        train_step_end_for_fraction(make_step_counts([100]), 0.8)


def _scores(y, signal, rng):
    return y * signal + rng.normal(size=len(y))


def test_generalization_gap_flags_overfit_model():
    rng = np.random.default_rng(0)
    y_train = (rng.random(20_000) < 0.01).astype(int)
    y_test = (rng.random(20_000) < 0.03).astype(int)
    # Strong separation in-sample, almost none held out.
    gap = generalization_gap(y_train, _scores(y_train, 4.0, rng), y_test, _scores(y_test, 0.2, rng))
    assert gap["roc_auc_gap"] > 0.05
    assert gap["overfit_warning"]


def test_generalization_gap_ok_when_performance_holds_despite_base_rate_shift():
    rng = np.random.default_rng(1)
    y_train = (rng.random(50_000) < 0.01).astype(int)
    y_test = (rng.random(50_000) < 0.03).astype(int)  # 3x base rate shift
    gap = generalization_gap(y_train, _scores(y_train, 2.0, rng), y_test, _scores(y_test, 2.0, rng))
    assert abs(gap["roc_auc_gap"]) < 0.02
    assert not gap["overfit_warning"]
