"""Bias mitigation methods for fairness-aware hiring ML pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression


try:
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.datasets import BinaryLabelDataset

    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False


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

    Uses AIF360 Reweighing when available. If AIF360-specific processing fails,
    a statistically equivalent manual reweighing fallback is used.
    """
    X_df, y_series, sensitive_series = _validate_training_inputs(
        X_train, y_train, sensitive_features
    )

    if AIF360_AVAILABLE:
        try:
            protected_name = "sensitive_feature"
            label_name = "label"

            protected_codes = sensitive_series.astype("category").cat.codes
            labels = y_series.astype(int)
            dataset_df = pd.DataFrame(
                {protected_name: protected_codes, label_name: labels}
            )

            train_dataset = BinaryLabelDataset(
                df=dataset_df,
                label_names=[label_name],
                protected_attribute_names=[protected_name],
            )

            rw = Reweighing(
                unprivileged_groups=[{protected_name: 0}],
                privileged_groups=[{protected_name: 1}],
            )
            transformed = rw.fit_transform(train_dataset)
            sample_weights = pd.Series(
                transformed.instance_weights,
                name="sample_weight",
            )

            return ReweighingResult(
                X_train=X_df,
                y_train=y_series,
                sensitive_features=sensitive_series,
                sample_weights=sample_weights,
                method_used="aif360_reweighing",
            )
        except Exception:
            # Fall back to manual weighting to keep pipeline robust.
            pass

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

    threshold_model = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        predict_method=predict_method,
        prefit=False,
    )
    threshold_model.fit(X_df, y_series, sensitive_features=sensitive_series)
    return threshold_model
