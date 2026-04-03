"""Correlation analysis utilities for proxy-bias diagnosis.

This module identifies potential proxy discrimination by measuring how strongly
candidate features correlate with a sensitive attribute (for example: gender).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_inputs(
    df: pd.DataFrame,
    sensitive_attribute: str,
) -> None:
    """Validate dataset and required sensitive attribute column."""
    if df is None or df.empty:
        raise ValueError("Input dataframe must be non-empty.")

    if sensitive_attribute not in df.columns:
        raise ValueError(
            f"Sensitive attribute column '{sensitive_attribute}' not found in dataframe."
        )


def _encode_sensitive_attribute(series: pd.Series) -> pd.Series:
    """Encode sensitive attribute into numeric values for correlation math."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    categorical = series.fillna("Unknown").astype("category")
    return pd.Series(categorical.cat.codes, index=series.index, dtype=float)


def analyze_feature_correlation_with_sensitive_attribute(
    df: pd.DataFrame,
    sensitive_attribute: str,
    top_k: int = 5,
    method: str = "pearson",
) -> dict[str, list[dict[str, Any]]]:
    """Compute top absolute correlations between features and sensitive attribute.

    Args:
        df: Input hiring dataset.
        sensitive_attribute: Protected column name (for example: gender).
        top_k: Number of top correlated features to return.
        method: Correlation method supported by pandas (pearson, spearman, kendall).

    Returns:
        Dictionary in reporting-friendly format:
        {
            "top_correlated_features": [
                {"feature": "university_rank", "correlation": 0.72},
                {"feature": "employment_gap", "correlation": 0.65}
            ]
        }
    """
    _validate_inputs(df, sensitive_attribute)

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    sensitive_encoded = _encode_sensitive_attribute(df[sensitive_attribute])

    correlations: list[dict[str, Any]] = []
    for feature in df.columns:
        if feature == sensitive_attribute:
            continue

        feature_series = df[feature]
        if not pd.api.types.is_numeric_dtype(feature_series):
            feature_series = feature_series.fillna(
                "Unknown").astype("category").cat.codes
        else:
            feature_series = pd.to_numeric(feature_series, errors="coerce")

        # Correlation can be NaN for constant columns or all-missing overlap.
        corr_value = feature_series.corr(sensitive_encoded, method=method)
        if corr_value is None or np.isnan(corr_value):
            continue

        correlations.append(
            {
                "feature": str(feature),
                "correlation": float(round(float(corr_value), 6)),
                "abs_correlation": float(abs(corr_value)),
            }
        )

    top_features = sorted(
        correlations,
        key=lambda item: item["abs_correlation"],
        reverse=True,
    )[:top_k]

    # Keep only the requested output schema.
    result_features = [
        {"feature": item["feature"], "correlation": item["correlation"]}
        for item in top_features
    ]

    return {"top_correlated_features": result_features}
