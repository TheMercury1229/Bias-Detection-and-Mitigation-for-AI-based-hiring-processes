"""Model evaluation utilities for baseline and fairness-mitigated hiring models."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Suppress sklearn warnings for class imbalance
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from src.bias_detection.fairness_metrices import compute_fairness_metrics
except ImportError:
    try:
        from src.bias_detection.fairness_metrics import compute_fairness_metrics
    except ImportError:
        from ..bias_detection.fairness_metrices import compute_fairness_metrics


def _to_numpy_1d(values: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    """Convert values to flattened numpy array."""
    return np.asarray(values).ravel()


def _validate_eval_inputs(
        y_true: pd.Series | np.ndarray | list[Any],
        y_pred: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate prediction, label, and sensitive-feature arrays."""
    y_true_arr = _to_numpy_1d(y_true)
    y_pred_arr = _to_numpy_1d(y_pred)
    sensitive_arr = _to_numpy_1d(sensitive_features)

    if y_true_arr.size == 0 or y_pred_arr.size == 0 or sensitive_arr.size == 0:
        raise ValueError("Evaluation inputs must be non-empty.")

    if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_arr)):
        raise ValueError(
            "Input length mismatch: y_true, y_pred, and sensitive_features must "
            "have the same length."
        )

    return y_true_arr, y_pred_arr, sensitive_arr


def compute_performance_metrics(
        y_true: pd.Series | np.ndarray | list[Any],
        y_pred: pd.Series | np.ndarray | list[Any],
) -> dict[str, Any]:
    """Compute core classification metrics for hiring prediction evaluation."""
    y_true_arr = _to_numpy_1d(y_true)
    y_pred_arr = _to_numpy_1d(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    # Prefer binary metrics, then gracefully fallback for multiclass outputs.
    try:
        precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    except ValueError:
        precision = precision_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
        )

    matrix = confusion_matrix(y_true_arr, y_pred_arr)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_predictions(
        y_true: pd.Series | np.ndarray | list[Any],
        y_pred: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
        model_name: str = "model",
) -> dict[str, Any]:
    """Evaluate predictive performance and fairness metrics for one model."""
    y_true_arr, y_pred_arr, sensitive_arr = _validate_eval_inputs(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    performance_metrics = compute_performance_metrics(y_true_arr, y_pred_arr)
    fairness_metrics = compute_fairness_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        sensitive_features=sensitive_arr,
    )

    return {
        "model_name": model_name,
        "performance": performance_metrics,
        "fairness": fairness_metrics,
    }


def compare_baseline_and_mitigated_models(
        y_true: pd.Series | np.ndarray | list[Any],
        y_pred_baseline: pd.Series | np.ndarray | list[Any],
        y_pred_mitigated: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
        baseline_name: str = "baseline",
        mitigated_name: str = "fairness_mitigated",
) -> dict[str, Any]:
    """Compare baseline vs fairness-mitigated model outputs.

    Returns a structured dictionary suitable for dashboard visualizations.
    """
    baseline_result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred_baseline,
        sensitive_features=sensitive_features,
        model_name=baseline_name,
    )
    mitigated_result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred_mitigated,
        sensitive_features=sensitive_features,
        model_name=mitigated_name,
    )

    delta = {
        "accuracy_change": (
            mitigated_result["performance"]["accuracy"]
            - baseline_result["performance"]["accuracy"]
        ),
        "precision_change": (
            mitigated_result["performance"]["precision"]
            - baseline_result["performance"]["precision"]
        ),
        "recall_change": (
            mitigated_result["performance"]["recall"]
            - baseline_result["performance"]["recall"]
        ),
        "demographic_parity_difference_change": (
            mitigated_result["fairness"]["demographic_parity_difference"]
            - baseline_result["fairness"]["demographic_parity_difference"]
        ),
        "equalized_odds_difference_change": (
            mitigated_result["fairness"]["equalized_odds_difference"]
            - baseline_result["fairness"]["equalized_odds_difference"]
        ),
        "disparate_impact_ratio_change": (
            mitigated_result["fairness"]["disparate_impact_ratio"]
            - baseline_result["fairness"]["disparate_impact_ratio"]
        ),
    }

    return {
        "baseline": baseline_result,
        "mitigated": mitigated_result,
        "delta": delta,
    }
