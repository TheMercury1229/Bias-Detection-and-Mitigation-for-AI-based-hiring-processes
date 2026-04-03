"""Group-level SHAP comparison utilities for fairness analysis.

This module compares average feature contributions across demographic groups so
the pipeline can detect whether the model treats groups differently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.explainability.shap_explainer import compute_shap_values


def _validate_inputs(
    X: pd.DataFrame | np.ndarray,
    sensitive_features: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and normalize feature matrix and sensitive attribute values."""
    if X is None:
        raise ValueError("X must not be None.")
    if sensitive_features is None:
        raise ValueError("sensitive_features must not be None.")

    if isinstance(X, pd.DataFrame):
        X_df = X.reset_index(drop=True)
    else:
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        X_df = pd.DataFrame(X_array)

    if isinstance(sensitive_features, pd.Series):
        sensitive_series = sensitive_features.reset_index(drop=True)
    else:
        sensitive_series = pd.Series(np.asarray(sensitive_features).ravel())

    if X_df.empty:
        raise ValueError("X must contain at least one row.")
    if sensitive_series.empty:
        raise ValueError("sensitive_features must contain at least one row.")
    if len(X_df) != len(sensitive_series):
        raise ValueError(
            "X and sensitive_features must have the same number of rows."
        )

    return X_df, sensitive_series.astype(str)


def _coerce_group_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize SHAP outputs into a 2D numpy array."""
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Defensive fallback for unexpected SHAP output shapes.
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError("SHAP values must be a 2D array after selection.")
    return values


def analyze_group_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    sensitive_features: pd.Series | np.ndarray,
    *,
    feature_names: list[str] | None = None,
    class_index: int | None = None,
    use_absolute_values: bool = True,
) -> dict[str, dict[str, float]]:
    """Compare average SHAP feature importance across sensitive groups.

    Args:
        model: A fitted tree-based model.
        X: Feature matrix used for explanation.
        sensitive_features: Group labels for each row, such as gender.
        feature_names: Optional explicit feature names for array inputs.
        class_index: Optional class index forwarded to SHAP for multiclass models.
        use_absolute_values: If True, average absolute SHAP values per group.

    Returns:
        Dictionary mapping each group to a feature-importance dictionary.
        Example:
        {
            "Male": {"experience": 0.25, "university_rank": 0.30},
            "Female": {"experience": 0.20, "employment_gap": 0.35}
        }
    """
    X_df, sensitive_series = _validate_inputs(
        X=X, sensitive_features=sensitive_features)

    shap_result = compute_shap_values(
        model=model,
        X=X_df,
        feature_names=feature_names,
        class_index=class_index,
    )
    shap_values = _coerce_group_shap_values(shap_result["shap_values"])
    feature_labels = list(shap_result["feature_names"])

    if shap_values.shape[0] != len(X_df):
        raise ValueError("SHAP values must align with the rows in X.")
    if shap_values.shape[1] != len(feature_labels):
        raise ValueError(
            "SHAP values must align with the feature dimension of X.")

    if use_absolute_values:
        shap_values = np.abs(shap_values)

    comparison: dict[str, dict[str, float]] = {}
    for group_name in sorted(sensitive_series.dropna().unique().tolist()):
        group_mask = sensitive_series == group_name
        group_values = shap_values[group_mask.to_numpy()]

        if group_values.size == 0:
            continue

        group_means = group_values.mean(axis=0)
        comparison[str(group_name)] = {
            str(feature_labels[idx]): float(round(value, 6))
            for idx, value in enumerate(group_means)
        }

    return comparison
