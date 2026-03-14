"""Data preprocessing utilities for fairness-aware hiring pipelines."""

from __future__ import annotations
from typing import Iterable
import pandas as pd
from sklearn.model_selection import train_test_split


def validate_non_empty_dataset(df: pd.DataFrame) -> None:
    """Raise an error when the input dataset is empty."""
    if df is None:
        raise ValueError("Dataset is None. Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty. Provide a non-empty dataset.")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in a dataframe.

    Numeric columns are filled with median and categorical columns with mode.
    """
    processed = df.copy()

    numeric_columns = processed.select_dtypes(include=["number"]).columns
    categorical_columns = processed.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for col in numeric_columns:
        median_value = processed[col].median()
        if pd.isna(median_value):
            median_value = 0
        processed[col] = processed[col].fillna(median_value)

    for col in categorical_columns:
        mode_series = processed[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "missing"
        processed[col] = processed[col].fillna(fill_value)

    return processed


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features using pandas."""
    return pd.get_dummies(df, drop_first=False)


def resolve_sensitive_attribute(
        df: pd.DataFrame, sensitive_attributes: str | Iterable[str]
) -> str:
    """Resolve and validate the sensitive attribute column.

    Supports either a single column name or multiple candidates.
    If multiple are provided, the first present column is selected.
    """
    if isinstance(sensitive_attributes, str):
        candidates = [sensitive_attributes]
    else:
        candidates = list(sensitive_attributes)

    if not candidates:
        raise ValueError("At least one sensitive attribute must be provided.")

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No sensitive attribute column found. Expected at least one of: "
        + ", ".join(candidates)
    )


def preprocess_dataset(
        df: pd.DataFrame,
        target_column: str,
        sensitive_attributes: str | Iterable[str],
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_target: bool = True,
        include_sensitive_in_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Preprocess dataset for model training and fairness analysis.

    Steps:
    1. Validate dataset and required columns.
    2. Handle missing values.
    3. Identify sensitive attribute.
    4. Separate features and target.
    5. One-hot encode categorical features.
    6. Split into train/test sets and return sensitive features separately.
    """
    validate_non_empty_dataset(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from the dataset.")

    sensitive_column = resolve_sensitive_attribute(df, sensitive_attributes)

    processed = fill_missing_values(df)

    y = processed[target_column]
    sensitive_feature = processed[sensitive_column]

    feature_columns = [
        col for col in processed.columns if col != target_column]
    if not include_sensitive_in_features:
        feature_columns = [
            col for col in feature_columns if col != sensitive_column]

    X = processed[feature_columns]
    X = encode_categorical_features(X)

    stratify = y if stratify_target and y.nunique(dropna=False) > 1 else None

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_feature_train,
        sensitive_feature_test,
    ) = train_test_split(
        X,
        y,
        sensitive_feature,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_feature_train,
        sensitive_feature_test,
    )
