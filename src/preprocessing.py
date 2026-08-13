"""
Reusable feature engineering and preprocessing functions for the
credit risk dataset. Import these into the notebook and into app.py
so training and inference use identical logic.

Adjust column names in `engineer_features` to match whichever raw
dataset you download (LendingClub vs Give Me Some Credit have
different schemas).
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that are known to be predictive for credit risk.
    Expects raw columns like: annual_income, monthly_debt, revolving_balance,
    revolving_limit, employment_length_years, loan_amount. Rename/adjust
    to match your chosen dataset's actual column names.
    """
    df = df.copy()

    if {"monthly_debt", "annual_income"}.issubset(df.columns):
        monthly_income = df["annual_income"] / 12
        df["debt_to_income"] = df["monthly_debt"] / monthly_income.replace(0, np.nan)

    if {"revolving_balance", "revolving_limit"}.issubset(df.columns):
        df["credit_utilization"] = (
            df["revolving_balance"] / df["revolving_limit"].replace(0, np.nan)
        )

    if "employment_length_years" in df.columns:
        df["employment_bucket"] = pd.cut(
            df["employment_length_years"],
            bins=[-1, 1, 3, 7, np.inf],
            labels=["<1yr", "1-3yrs", "3-7yrs", "7+yrs"],
        )

    if {"loan_amount", "annual_income"}.issubset(df.columns):
        df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)

    return df


def handle_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute numeric columns; fill categorical NaNs with 'missing'."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)

    for col in categorical_cols:
        df[col] = df[col].fillna("missing")

    return df


def encode_categoricals(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    if columns is None:
        columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return pd.get_dummies(df, columns=columns, drop_first=True)
