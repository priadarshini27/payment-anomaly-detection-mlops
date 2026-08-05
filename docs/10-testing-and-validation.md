# 10 — Testing & Validation

[← 09 — Dashboard](09-dashboard.md) | [Index](README.md) | Next: [11 — Deployment & Operations →](11-deployment-and-operations.md)

## Test suite

Unit tests cover the three areas with the most business logic and the
highest cost-of-being-wrong: feature engineering, data cleaning rules, and
the drift-detection math. Tests intentionally avoid depending on the full
6.36M-row dataset or a populated database — they construct small,
hand-crafted DataFrames so they run in seconds and pin down exact
expected values.

### File: pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

### File: tests/__init__.py

```python
```

(Empty — present only so `tests` is treated as a package.)

### File: tests/test_feature_engineering.py

```python
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
```

### File: tests/test_clean_data.py

```python
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
```

### File: tests/test_drift.py

```python
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
```

## Running the tests

```bash
python -m pytest -q
```

Measured result on this project (24 tests, no database or full dataset
required):

```
........................                                                 [100%]
24 passed in 19.89s
```

## Manual end-to-end validation checklist

After running the full pipeline (see
[01-getting-started.md](01-getting-started.md), Step 4), validate the
result manually:

```bash
sqlite3 database/database.db
```

```sql
-- Row count should equal 6,362,604 (6,362,620 raw minus 16 zero-amount rows)
SELECT COUNT(*) FROM transactions;

-- Fraud count should be 8,213 (isFraud is never modified by cleaning)
-- minus however many of the 16 dropped zero-amount rows were fraud (all 16 were)
-- => expect 8,213 - 16 = 8,197
SELECT SUM(isFraud) FROM transactions;

-- Every row should have velocity features populated after sql/queries.py runs
SELECT COUNT(*) FROM transactions WHERE txnVelocity1h IS NULL;  -- expect 0

-- Exactly one model_runs row per training run, is_deployed=1 on exactly one
SELECT COUNT(*) FROM model_runs WHERE is_deployed = 1;  -- expect 1 (after evaluate.py)

-- PR-AUC should be well above the 0.13% base rate implied by random guessing
SELECT run_id, pr_auc, roc_auc FROM model_runs ORDER BY run_id DESC LIMIT 1;

-- drift_metrics should have exactly 12 rows per drift-check run_id
-- (one per MODEL_FEATURE_COLUMNS feature)
SELECT run_id, COUNT(*) FROM drift_metrics GROUP BY run_id;
```

### Sanity ranges

| Metric | Expected range | Why |
|---|---|---|
| PR-AUC | Meaningfully above 0.0013 (the base fraud rate) | A model with zero discriminative power would score at the base rate |
| ROC-AUC | > 0.5 | Anything at or below 0.5 means the model is no better than random (or worse) |
| Precision@0.5% | Well above 0.0013 | Confirms the top-scored transactions are enriched for fraud relative to the base rate |
| PSI values on a same-distribution check | All < 0.10 | Comparing a window against itself (or a very similar one) should show no drift |

---
[← 09 — Dashboard](09-dashboard.md) | [Index](README.md) | Next: [11 — Deployment & Operations →](11-deployment-and-operations.md)
