"""SHAP explainability utilities for hiring model analysis.

This module provides reusable helpers for computing SHAP values on tree-based
models so the results can be used for bias analysis and visualization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover - depends on environment
    shap = None  # type: ignore[assignment]
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None


def _validate_inputs(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Validate model and feature matrix inputs."""
    if shap is None:
        raise ImportError(
            "The 'shap' package is required for SHAP explanations."
        ) from _SHAP_IMPORT_ERROR

    if X is None:
        raise ValueError("X must not be None.")

    if isinstance(X, pd.DataFrame):
        X_df = X.reset_index(drop=True)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        X_df = pd.DataFrame(array)

    if X_df.empty:
        raise ValueError("X must contain at least one row.")

    if not hasattr(model, "predict"):
        raise ValueError(
            "model must be a fitted estimator with a predict method.")

    return X_df


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
    class_index: int | None = None,
) -> dict[str, Any]:
    """Compute SHAP values for a dataset using a tree-based model.

    Args:
        model: A fitted tree-based model, such as RandomForestClassifier.
        X: Feature matrix used for explanation.
        feature_names: Optional explicit feature names for array inputs.
        class_index: Optional class index to select for multi-class outputs.

    Returns:
        Dictionary with SHAP values and supporting metadata:
        {
            "shap_values": ...,
            "expected_value": ...,
            "feature_names": [...],
            "model_type": "RandomForestClassifier"
        }
    """
    X_df = _validate_inputs(model=model, X=X)

    if feature_names is not None:
        if len(feature_names) != X_df.shape[1]:
            raise ValueError(
                "feature_names must match the number of columns in X."
            )
        X_df.columns = list(feature_names)

    if not hasattr(model, "estimators_"):
        raise ValueError(
            "compute_shap_values currently supports tree-based models only."
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    expected_value = explainer.expected_value

    # For binary/multiclass classifiers, SHAP may return a list of arrays.
    if isinstance(shap_values, list):
        if class_index is None:
            class_index = 1 if len(shap_values) > 1 else 0
        if class_index < 0 or class_index >= len(shap_values):
            raise ValueError(
                f"class_index must be between 0 and {len(shap_values) - 1}."
            )
        selected_shap_values = shap_values[class_index]
        selected_expected_value = (
            expected_value[class_index]
            if isinstance(expected_value, (list, np.ndarray))
            else expected_value
        )
    else:
        values = np.asarray(shap_values)
        if values.ndim == 2:
            selected_shap_values = values
            selected_expected_value = expected_value
        elif values.ndim == 3:
            # Handle newer SHAP formats where class dimension is part of ndarray.
            # Common shape for classifiers: (n_samples, n_features, n_classes).
            if values.shape[0] == X_df.shape[0] and values.shape[1] == X_df.shape[1]:
                n_classes = values.shape[2]
                idx = (1 if n_classes >
                       1 else 0) if class_index is None else class_index
                if idx < 0 or idx >= n_classes:
                    raise ValueError(
                        f"class_index must be between 0 and {n_classes - 1}."
                    )
                selected_shap_values = values[:, :, idx]
                if isinstance(expected_value, (list, np.ndarray)) and len(np.asarray(expected_value).shape) > 0:
                    selected_expected_value = np.asarray(
                        expected_value).ravel()[idx]
                else:
                    selected_expected_value = expected_value
            # Fallback shape occasionally seen: (n_classes, n_samples, n_features).
            elif values.shape[1] == X_df.shape[0] and values.shape[2] == X_df.shape[1]:
                n_classes = values.shape[0]
                idx = (1 if n_classes >
                       1 else 0) if class_index is None else class_index
                if idx < 0 or idx >= n_classes:
                    raise ValueError(
                        f"class_index must be between 0 and {n_classes - 1}."
                    )
                selected_shap_values = values[idx]
                if isinstance(expected_value, (list, np.ndarray)) and len(np.asarray(expected_value).shape) > 0:
                    selected_expected_value = np.asarray(
                        expected_value).ravel()[idx]
                else:
                    selected_expected_value = expected_value
            else:
                raise ValueError(
                    f"Unsupported SHAP array shape {values.shape}."
                )
        else:
            raise ValueError(
                f"Unsupported SHAP output dimension: {values.ndim}."
            )

    return {
        "shap_values": selected_shap_values,
        "expected_value": selected_expected_value,
        "feature_names": list(X_df.columns),
        "model_type": type(model).__name__,
        "n_samples": int(X_df.shape[0]),
        "n_features": int(X_df.shape[1]),
    }
