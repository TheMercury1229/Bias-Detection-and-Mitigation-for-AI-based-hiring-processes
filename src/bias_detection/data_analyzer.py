"""Dataset analysis utilities for fairness-aware bias diagnosis.

This module focuses on pre-model diagnostics by summarizing protected-group
representation and observed selection outcomes in historical data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _validate_columns(
    df: pd.DataFrame,
    sensitive_attribute: str,
    target_column: str,
) -> None:
    """Validate required columns and basic dataset conditions."""
    if df is None or df.empty:
        raise ValueError("Input dataframe must be non-empty.")

    missing = [
        col
        for col in (sensitive_attribute, target_column)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _to_binary_target(
    target_series: pd.Series,
    positive_label: Any,
) -> pd.Series:
    """Convert target values into 0/1 indicators for selection-rate math."""
    return (target_series == positive_label).astype(int)


def _infer_positive_label(target_series: pd.Series) -> Any:
    """Infer a reasonable positive label for binary target columns."""
    values = target_series.dropna().unique().tolist()
    if len(values) != 2:
        return 1

    if set(values).issubset({0, 1}):
        return 1
    if set(values).issubset({-1, 1}):
        return 1

    normalized = {str(value).strip().lower(): value for value in values}
    for candidate in ["1", "true", "yes", "y", "hired", "selected", "positive"]:
        if candidate in normalized:
            return normalized[candidate]

    return sorted(values, key=lambda value: str(value).lower())[-1]


def analyze_group_distribution_and_selection_rate(
    df: pd.DataFrame,
    sensitive_attribute: str,
    target_column: str,
    positive_label: Any | None = None,
    fill_missing_group: str = "Unknown",
) -> dict[str, dict[str, float | int]]:
    """Analyze group distribution and selection rate for a protected attribute.

    Args:
        df: Input dataset containing protected attribute and target columns.
        sensitive_attribute: Protected column name (for example: gender).
        target_column: Outcome label column (for example: hired).
        positive_label: Value in target_column interpreted as selected/hired.
            If None, it is inferred for binary targets.
        fill_missing_group: Label used when sensitive attribute value is missing.

    Returns:
        Dictionary with keys:
        - group_distribution: raw count per sensitive group
        - selection_rate: positive outcome rate per sensitive group

    Example:
        {
            "group_distribution": {"Male": 600, "Female": 400},
            "selection_rate": {"Male": 0.65, "Female": 0.40}
        }
    """
    _validate_columns(df, sensitive_attribute, target_column)

    groups = df[sensitive_attribute].fillna(fill_missing_group).astype(str)
    if positive_label is None:
        positive_label = _infer_positive_label(df[target_column])
    y_selected = _to_binary_target(df[target_column], positive_label)

    group_distribution = groups.value_counts(dropna=False).to_dict()
    selection_rate = y_selected.groupby(groups).mean().to_dict()

    return {
        "group_distribution": {
            str(group): int(count)
            for group, count in group_distribution.items()
        },
        "selection_rate": {
            str(group): float(round(rate, 6))
            for group, rate in selection_rate.items()
        },
    }
