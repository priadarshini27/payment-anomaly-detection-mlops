"""Unit tests for preprocessing/clean_data.py."""

import pandas as pd
import pytest

from preprocessing.clean_data import (
    RAW_COLUMNS,
    SchemaValidationError,
    clean,
    drop_duplicates,
    standardize_categoricals,
    validate_amounts,
    validate_schema,
)


def make_raw_df(n_dupes=0):
    rows = [
        {
            "step": 1, "type": "PAYMENT", "amount": 100.0, "nameOrig": "C1",
            "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0, "nameDest": "M1",
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFraud": 0, "isFlaggedFraud": 0,
        },
        {
            "step": 2, "type": "TRANSFER", "amount": 0.0, "nameOrig": "C2",
            "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0, "nameDest": "C3",
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFraud": 1, "isFlaggedFraud": 0,
        },
        {
            "step": 3, "type": "cash_out", "amount": 250.0, "nameOrig": "C4",
            "oldbalanceOrg": 250.0, "newbalanceOrig": 0.0, "nameDest": "C5",
            "oldbalanceDest": 0.0, "newbalanceDest": 250.0, "isFraud": 0, "isFlaggedFraud": 0,
        },
    ]
    df = pd.DataFrame(rows)
    for _ in range(n_dupes):
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_validate_schema_accepts_correct_columns():
    df = make_raw_df()
    validate_schema(df)  # should not raise


def test_validate_schema_rejects_missing_column():
    df = make_raw_df().drop(columns=["isFraud"])
    with pytest.raises(SchemaValidationError):
        validate_schema(df)


def test_validate_schema_rejects_extra_column():
    df = make_raw_df()
    df["unexpected"] = 1
    with pytest.raises(SchemaValidationError):
        validate_schema(df)


def test_drop_duplicates_removes_exact_duplicate_rows():
    df = make_raw_df(n_dupes=2)
    out, n_removed = drop_duplicates(df)
    assert n_removed == 2
    assert len(out) == 3


def test_validate_amounts_drops_zero_amount_rows():
    df = make_raw_df()
    out, n_removed = validate_amounts(df)
    assert n_removed == 1  # the single amount=0 row
    assert (out["amount"] > 0).all()


def test_standardize_categoricals_upper_cases_type():
    df = make_raw_df()
    out = standardize_categoricals(df)
    assert set(out["type"].astype(str)) == {"PAYMENT", "TRANSFER", "CASH_OUT"}


def test_clean_end_to_end_produces_engineered_columns_and_stats():
    df = make_raw_df()
    cleaned, stats = clean(df)
    assert stats["rows_in"] == 3
    assert stats["invalid_amount_rows_removed"] == 1
    assert stats["rows_out"] == 2
    assert "logAmount" in cleaned.columns
    assert "balanceDeltaOrig" in cleaned.columns
    assert "type_TRANSFER" in cleaned.columns
