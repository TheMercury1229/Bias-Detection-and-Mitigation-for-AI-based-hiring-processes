"""Bias mitigation methods for fairness-aware hiring ML pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ReweighingResult:
    """Container for reweighing outputs."""

    X_train: pd.DataFrame
    y_train: pd.Series
    sensitive_features: pd.Series
    sample_weights: pd.Series
    method_used: str


def _to_series(values: pd.Series | np.ndarray | list[Any], name: str) -> pd.Series:
    """Convert supported input types to a pandas Series."""
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    array = np.asarray(values).ravel()
    return pd.Series(array, name=name)


def _to_dataframe(values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Convert supported feature matrix inputs to a pandas DataFrame."""
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True)
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("X_train must be 2D (rows x features).")
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _validate_training_inputs(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate and normalize mitigation inputs."""
    X_df = _to_dataframe(X_train)
    y_series = _to_series(y_train, name="label")
    sensitive_series = _to_series(sensitive_features, name="sensitive_feature")

    if X_df.empty:
        raise ValueError("X_train is empty.")
    if y_series.empty or sensitive_series.empty:
        raise ValueError("y_train and sensitive_features must be non-empty.")
    if not (len(X_df) == len(y_series) == len(sensitive_series)):
        raise ValueError(
            "Input length mismatch: X_train, y_train, and sensitive_features must "
            "have the same number of rows."
        )

    return X_df, y_series, sensitive_series


def _ensure_binary_zero_one_labels(y_series: pd.Series, context: str) -> pd.Series:
    """Ensure labels are binary {0,1} for Fairlearn mitigation algorithms."""
    unique_values = sorted(pd.Series(y_series).dropna().unique().tolist())
    if set(unique_values).issubset({0, 1}):
        return y_series.astype(int)

    raise ValueError(
        f"{context} requires binary labels encoded as 0/1, but found labels: {unique_values}. "
        "Choose a binary target column or preprocess labels to 0/1 before fairness mitigation."
    )


def _manual_reweighing(
        y_series: pd.Series,
        sensitive_series: pd.Series,
) -> pd.Series:
    """Fallback reweighing based on P(A)P(Y)/P(A,Y)."""
    total = len(y_series)
    group_counts = sensitive_series.value_counts(dropna=False)
    label_counts = y_series.value_counts(dropna=False)
    joint_counts = (
        pd.DataFrame({"group": sensitive_series, "label": y_series})
        .value_counts(dropna=False)
        .rename("count")
    )

    weights = []
    for group, label in zip(sensitive_series, y_series):
        p_group = group_counts[group] / total
        p_label = label_counts[label] / total
        p_joint = joint_counts[(group, label)] / total
        weight = (p_group * p_label) / p_joint if p_joint > 0 else 1.0
        weights.append(float(weight))

    return pd.Series(weights, name="sample_weight")


def apply_data_reweighing(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
) -> ReweighingResult:
    """Apply data reweighing as a preprocessing mitigation method.

    Uses manual reweighing (statistically equivalent to AIF360 but memory-efficient).
    This avoids memory overhead from AIF360 BinaryLabelDataset for large datasets.
    """
    X_df, y_series, sensitive_series = _validate_training_inputs(
        X_train, y_train, sensitive_features
    )

    # Use memory-efficient manual reweighing directly
    # This is statistically equivalent to AIF360 but avoids data duplication
    sample_weights = _manual_reweighing(
        y_series=y_series, sensitive_series=sensitive_series)

    return ReweighingResult(
        X_train=X_df,
        y_train=y_series,
        sensitive_features=sensitive_series,
        sample_weights=sample_weights,
        method_used="manual_reweighing",
    )


def train_fairness_constrained_model(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
        constraint: str = "demographic_parity",
        base_estimator: Any | None = None,
        random_state: int = 42,
) -> ExponentiatedGradient:
    """Train a fairness-constrained model using Fairlearn reductions."""
    X_df, y_series, sensitive_series = _validate_training_inputs(
        X_train, y_train, sensitive_features
    )
    y_series = _ensure_binary_zero_one_labels(
        y_series,
        context="Fairness-Constrained Learning",
    )

    if base_estimator is None:
        base_estimator = LogisticRegression(
            max_iter=1000, random_state=random_state)

    constraint_key = constraint.strip().lower()
    if constraint_key == "demographic_parity":
        fairness_constraint = DemographicParity()
    elif constraint_key == "equalized_odds":
        fairness_constraint = EqualizedOdds()
    else:
        raise ValueError(
            "Unsupported constraint. Use 'demographic_parity' or 'equalized_odds'."
        )

    model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=fairness_constraint,
    )
    model.fit(X_df, y_series, sensitive_features=sensitive_series)
    return model


def train_threshold_optimizer(
        estimator: Any,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        sensitive_features: pd.Series | np.ndarray | list[Any],
        constraints: str = "demographic_parity",
        predict_method: str = "predict_proba",
) -> ThresholdOptimizer:
    """Fit Fairlearn threshold optimization as post-processing mitigation."""
    X_df, y_series, sensitive_series = _validate_training_inputs(
        X_train, y_train, sensitive_features
    )
    y_series = _ensure_binary_zero_one_labels(
        y_series,
        context="Post-processing threshold optimization",
    )

    threshold_model = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        predict_method=predict_method,
        prefit=False,
    )
    threshold_model.fit(X_df, y_series, sensitive_features=sensitive_series)
    return threshold_model
