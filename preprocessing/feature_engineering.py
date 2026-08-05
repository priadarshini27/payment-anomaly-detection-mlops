"""
preprocessing/feature_engineering.py
=====================================
Derives the engineered features used throughout the project (modeling,
drift monitoring, dashboard) from the raw/cleaned PaySim columns.

These functions are the single source of truth for feature derivation and
are called from two places:
  * preprocessing/clean_data.py -- applied once, in bulk, to the full
    historical dataset before it is loaded into SQLite.
  * models/predict.py -- applied to new/incoming transactions at scoring
    time, so a live transaction is featurized identically to how the
    training data was featurized.

Engineered features (see Document 1, Section 4.1):
  balanceDeltaOrig -- oldbalanceOrg - newbalanceOrig - amount. A non-zero
      residual means the originator's balance did not move by exactly the
      transacted amount -- PaySim's fraud generator introduces this
      deliberately, so it is a strong signal rather than noise to clean away.
  balanceDeltaDest -- oldbalanceDest + amount - newbalanceDest. Same idea,
      applied to the recipient side.
  origZeroedOut -- 1 if oldbalanceOrg > 0 and newbalanceOrig == 0, else 0.
      Captures full-balance drains typical of account-takeover fraud.
  logAmount -- log1p(amount). Variance-stabilizing transform since `amount`
      is heavily right-skewed.
  isMerchantDest -- 1 if nameDest starts with 'M' (a merchant account),
      else 0. Merchant accounts don't carry a real balance in PaySim
      (oldbalanceDest/newbalanceDest are structurally 0), which is why
      balanceDeltaDest behaves differently for merchant destinations.
  type_<CATEGORY> -- one-hot encoding of the five PaySim transaction types
      (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), since fraud is
      concentrated almost entirely in TRANSFER and CASH_OUT.

txnVelocity1h / txnVelocity24h are *not* computed here -- they require a
window function over the full ordered transaction history (per nameOrig,
by step) rather than a per-row/per-batch computation, so they are computed
once after load, in sql/queries.py::get_velocity_features().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# The feature set the Isolation Forest is trained and scored on. Deliberately
# excludes isFraud/isFlaggedFraud (label leakage) and identifier columns.
MODEL_FEATURE_COLUMNS = [
    "logAmount",
    "balanceDeltaOrig",
    "balanceDeltaDest",
    "origZeroedOut",
    "isMerchantDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "txnVelocity1h",
    "txnVelocity24h",
]


def add_balance_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds balanceDeltaOrig and balanceDeltaDest residual columns."""
    df = df.copy()
    df["balanceDeltaOrig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    ).astype("float32")
    df["balanceDeltaDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).astype("float32")
    return df


def add_origin_zeroed_out_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the binary origZeroedOut account-drain indicator."""
    df = df.copy()
    df["origZeroedOut"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype("int8")
    return df


def add_log_amount_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds logAmount = log1p(amount), a variance-stabilizing transform."""
    df = df.copy()
    df["logAmount"] = np.log1p(df["amount"]).astype("float32")
    return df


def add_merchant_dest_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds isMerchantDest = 1 if nameDest is a merchant account (prefix 'M')."""
    df = df.copy()
    df["isMerchantDest"] = df["nameDest"].astype(str).str.startswith("M").astype("int8")
    return df


def add_type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one-hot columns type_CASH_IN .. type_TRANSFER for the 5 PaySim types."""
    df = df.copy()
    type_str = df["type"].astype(str).str.upper()
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (type_str == t).astype("int8")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies every engineered feature in the documented order. Idempotent:
    calling it twice on an already-engineered frame recomputes the same
    values rather than stacking columns.

    Does NOT compute txnVelocity1h/24h -- see module docstring.
    """
    df = add_balance_delta_features(df)
    df = add_origin_zeroed_out_feature(df)
    df = add_log_amount_feature(df)
    df = add_merchant_dest_feature(df)
    df = add_type_one_hot(df)
    return df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and orders the columns the Isolation Forest is trained/scored
    on (MODEL_FEATURE_COLUMNS), assuming engineer_features() and the
    velocity backfill have already run. Missing velocity columns (e.g. for
    a single newly-scored transaction with no history) default to 0.
    """
    df = df.copy()
    for col in ("txnVelocity1h", "txnVelocity24h"):
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"build_model_matrix: missing required feature columns {missing}. "
            "Did you call engineer_features() first?"
        )
    return df[MODEL_FEATURE_COLUMNS].astype("float64")
