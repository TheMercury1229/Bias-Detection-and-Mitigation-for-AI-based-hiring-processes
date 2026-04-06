"""Fairness metric utilities for hiring model evaluation.
This module provides reusable functions to compute group fairness metrics
using Fairlearn and return them in reporting-friendly dictionary format.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)


def _to_numpy_1d(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert supported array-like inputs to a flattened numpy array."""
    array = np.asarray(values)
    return array.ravel()


def _validate_inputs(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        sensitive_features: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize input arrays for fairness computations."""
    y_true_arr = _to_numpy_1d(y_true)
    y_pred_arr = _to_numpy_1d(y_pred)
    sensitive_arr = _to_numpy_1d(sensitive_features)

    if y_true_arr.size == 0 or y_pred_arr.size == 0 or sensitive_arr.size == 0:
        raise ValueError("Inputs must be non-empty arrays or pandas series.")

    if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_arr)):
        raise ValueError(
            "Input length mismatch: y_true, y_pred, and sensitive_features must "
            "have the same number of rows."
        )

    return y_true_arr, y_pred_arr, sensitive_arr


def _infer_pos_label(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> Any:
    """Infer a stable positive label for binary classification metrics.

    Fairlearn's equalized_odds_difference relies on binary metrics that require
    a valid pos_label. This helper supports labels like {"Yes", "No"},
    {True, False}, {"1", "0"}, and numeric binaries.
    """
    combined = np.concatenate([y_true_arr, y_pred_arr])
    unique = pd.unique(pd.Series(combined).dropna())

    if len(unique) != 2:
        return 1

    label_set = set(unique.tolist())

    preferred_positive = [1, True, "1", "true",
                          "yes", "y", "hired", "positive"]
    normalized_to_original: dict[str, Any] = {
        str(v).strip().lower(): v for v in unique}
    for candidate in preferred_positive:
        key = str(candidate).strip().lower()
        if key in normalized_to_original:
            return normalized_to_original[key]

    try:
        ordered = sorted(unique.tolist(), key=lambda value: str(value).lower())
        return ordered[-1]
    except Exception:
        return list(label_set)[-1]


def _binarize_labels(
        y_true_arr: np.ndarray,
        y_pred_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert binary labels to {0,1} for compatibility across metric versions."""
    pos_label = _infer_pos_label(y_true_arr, y_pred_arr)
    y_true_bin = np.asarray(
        [1 if value == pos_label else 0 for value in y_true_arr])
    y_pred_bin = np.asarray(
        [1 if value == pos_label else 0 for value in y_pred_arr])
    return y_true_bin, y_pred_bin


def compute_selection_rate_by_group(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        sensitive_features: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute selection rate for each demographic group."""
    y_true_arr, y_pred_arr, sensitive_arr = _validate_inputs(
        y_true, y_pred, sensitive_features
    )

    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        sensitive_features=sensitive_arr,
    )

    group_rates: dict[str, float] = {}
    by_group = metric_frame.by_group.to_dict()
    for group, rate in by_group.items():
        group_rates[str(group)] = float(rate)

    return group_rates


def compute_disparate_impact_ratio(selection_rates_by_group: dict[str, float]) -> float:
    """Compute disparate impact ratio as min group selection rate / max group rate."""
    if not selection_rates_by_group:
        raise ValueError("selection_rates_by_group cannot be empty.")

    rates = [float(rate) for rate in selection_rates_by_group.values()]
    max_rate = max(rates)

    if max_rate == 0:
        # When no group receives positive predictions, DI is not informative.
        # Use neutral value to avoid falsely signaling severe selection bias.
        return 1.0

    return float(min(rates) / max_rate)


def compute_fairness_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        sensitive_features: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute fairness metrics for hiring model predictions.

    Args:
            y_true: Ground-truth labels (numpy array or pandas Series).
            y_pred: Model predictions (numpy array or pandas Series).
            sensitive_features: Sensitive attribute values for each row.

    Returns:
            Dictionary with:
                    - demographic_parity_difference
                    - equalized_odds_difference
                    - selection_rate_by_group
                    - disparate_impact_ratio
    """
    y_true_arr, y_pred_arr, sensitive_arr = _validate_inputs(
        y_true, y_pred, sensitive_features
    )

    y_true_bin, y_pred_bin = _binarize_labels(y_true_arr, y_pred_arr)

    dp_difference = demographic_parity_difference(
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        sensitive_features=sensitive_arr,
    )
    eo_difference = equalized_odds_difference(
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        sensitive_features=sensitive_arr,
    )

    selection_rates = compute_selection_rate_by_group(
        y_true=y_true_bin,
        y_pred=y_pred_bin,
        sensitive_features=sensitive_arr,
    )
    di_ratio = compute_disparate_impact_ratio(selection_rates)

    return {
        "demographic_parity_difference": float(dp_difference),
        "equalized_odds_difference": float(eo_difference),
        "selection_rate_by_group": selection_rates,
        "disparate_impact_ratio": float(di_ratio),
    }
