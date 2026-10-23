"""Unit tests for preprocessing/feature_engineering.py."""

import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    TRANSACTION_TYPES,
    add_balance_delta_features,
    add_log_amount_feature,
    add_merchant_dest_feature,
    add_origin_zeroed_out_feature,
    add_type_one_hot,
    build_model_matrix,
    engineer_features,
)


def make_sample_df():
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [100.0, 500.0, 0.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [1000.0, 500.0, 0.0],
            "newbalanceOrig": [900.0, 0.0, 0.0],
            "nameDest": ["M1", "C4", "C5"],
            "oldbalanceDest": [0.0, 200.0, 300.0],
            "newbalanceDest": [0.0, 700.0, 300.0],
        }
    )


def test_balance_delta_orig_is_zero_when_consistent():
    df = make_sample_df()
    out = add_balance_delta_features(df)
    # Row 0: 1000 - 900 - 100 = 0 (consistent)
    assert out.loc[0, "balanceDeltaOrig"] == pytest.approx(0.0)


def test_balance_delta_orig_nonzero_when_inconsistent():
    df = make_sample_df()
    out = add_balance_delta_features(df)
    # Row 2: oldbalanceOrg=0, newbalanceOrig=0, amount=0 -> residual is 0.
    # Row 1: 500 - 0 - 500 = 0 (consistent, full drain)
    assert out.loc[1, "balanceDeltaOrig"] == pytest.approx(0.0)


def test_origin_zeroed_out_flags_full_drain():
    df = make_sample_df()
    out = add_origin_zeroed_out_feature(df)
    # Row 1: oldbalanceOrg=500 > 0, newbalanceOrig=0 -> drained
    assert out.loc[1, "origZeroedOut"] == 1
    # Row 0: oldbalanceOrg=1000, newbalanceOrig=900 -> not drained
    assert out.loc[0, "origZeroedOut"] == 0
    # Row 2: oldbalanceOrg=0 -> not a drain (nothing to drain)
    assert out.loc[2, "origZeroedOut"] == 0


def test_log_amount_matches_log1p():
    df = make_sample_df()
    out = add_log_amount_feature(df)
    expected = np.log1p(df["amount"].to_numpy())
    np.testing.assert_allclose(out["logAmount"].to_numpy(), expected, rtol=1e-5)


def test_merchant_dest_prefix_detection():
    df = make_sample_df()
    out = add_merchant_dest_feature(df)
    assert out.loc[0, "isMerchantDest"] == 1  # nameDest = M1
    assert out.loc[1, "isMerchantDest"] == 0  # nameDest = C4


def test_type_one_hot_has_exactly_five_columns_summing_to_one():
    df = make_sample_df()
    out = add_type_one_hot(df)
    one_hot_cols = [f"type_{t}" for t in TRANSACTION_TYPES]
    assert set(one_hot_cols).issubset(out.columns)
    row_sums = out[one_hot_cols].sum(axis=1)
    assert (row_sums == 1).all()


def test_type_one_hot_case_insensitive():
    df = make_sample_df()
    df["type"] = df["type"].str.lower()
    out = add_type_one_hot(df)
    assert out.loc[0, "type_PAYMENT"] == 1


def test_engineer_features_is_idempotent():
    df = make_sample_df()
    once = engineer_features(df)
    twice = engineer_features(once)
    pd.testing.assert_frame_equal(
        once[["balanceDeltaOrig", "logAmount", "origZeroedOut"]],
        twice[["balanceDeltaOrig", "logAmount", "origZeroedOut"]],
    )


def test_build_model_matrix_raises_on_missing_columns():
    df = make_sample_df()  # not engineered yet
    with pytest.raises(KeyError):
        build_model_matrix(df)


def test_build_model_matrix_defaults_missing_velocity_to_zero():
    df = engineer_features(make_sample_df())
    matrix = build_model_matrix(df)
    assert list(matrix.columns) == MODEL_FEATURE_COLUMNS
    assert (matrix["txnVelocity1h"] == 0).all()
    assert (matrix["txnVelocity24h"] == 0).all()
