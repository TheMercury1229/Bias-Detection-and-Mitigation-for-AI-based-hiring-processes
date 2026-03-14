"""Training utilities for baseline and fairness-aware hiring models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _to_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Convert input features to a pandas DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("Training features must be a 2D matrix.")

    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _to_series(y: pd.Series | np.ndarray | list[Any], name: str = "target") -> pd.Series:
    """Convert labels to a pandas Series."""
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)

    return pd.Series(np.asarray(y).ravel(), name=name)


def _validate_training_data(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        sample_weight: pd.Series | np.ndarray | list[float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Validate and normalize training inputs."""
    X_df = _to_dataframe(X_train)
    y_series = _to_series(y_train)

    if X_df.empty:
        raise ValueError("X_train is empty.")
    if y_series.empty:
        raise ValueError("y_train is empty.")
    if len(X_df) != len(y_series):
        raise ValueError(
            "X_train and y_train must have the same number of rows.")

    sample_weight_series = None
    if sample_weight is not None:
        sample_weight_series = _to_series(sample_weight, name="sample_weight")
        if len(sample_weight_series) != len(X_df):
            raise ValueError(
                "sample_weight must have the same number of rows as X_train.")

    return X_df, y_series, sample_weight_series


def build_model(
        model_type: str = "logistic_regression",
        model_params: dict[str, Any] | None = None,
        random_state: int = 42,
) -> Any:
    """Build a scikit-learn model instance by name.

    Supported model_type values:
    - logistic_regression
    - random_forest
    """
    params = dict(model_params or {})
    key = model_type.strip().lower()

    if key in {"logistic_regression", "logistic", "lr"}:
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)

    if key in {"random_forest", "rf", "randomforest"}:
        params.setdefault("n_estimators", 200)
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)

    raise ValueError(
        "Unsupported model_type. Use 'logistic_regression' or 'random_forest'."
    )


def train_model(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        model_type: str = "logistic_regression",
        model_params: dict[str, Any] | None = None,
        sample_weight: pd.Series | np.ndarray | list[float] | None = None,
        random_state: int = 42,
) -> Any:
    """Train and return a single model.

    This function is reusable for both baseline and fairness-aware pipelines,
    because it accepts optional sample weights from mitigation methods.
    """
    X_df, y_series, sample_weight_series = _validate_training_data(
        X_train=X_train,
        y_train=y_train,
        sample_weight=sample_weight,
    )

    model = build_model(
        model_type=model_type,
        model_params=model_params,
        random_state=random_state,
    )

    fit_kwargs: dict[str, Any] = {}
    if sample_weight_series is not None:
        fit_kwargs["sample_weight"] = sample_weight_series

    model.fit(X_df, y_series, **fit_kwargs)
    return model


def train_baseline_model(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        model_type: str = "logistic_regression",
        model_params: dict[str, Any] | None = None,
        random_state: int = 42,
) -> Any:
    """Train and return a baseline model."""
    return train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        model_params=model_params,
        sample_weight=None,
        random_state=random_state,
    )


def train_multiple_models(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray | list[Any],
        model_configs: dict[str, dict[str, Any]] | None = None,
        sample_weight: pd.Series | np.ndarray | list[float] | None = None,
        random_state: int = 42,
) -> dict[str, Any]:
    """Train multiple models for comparison and return them in a dictionary.

    Example model_configs:
    {
            "logistic_baseline": {"model_type": "logistic_regression", "model_params": {}},
            "rf_baseline": {"model_type": "random_forest", "model_params": {"n_estimators": 300}}
    }
    """
    configs = model_configs or {
        "logistic_regression": {"model_type": "logistic_regression", "model_params": {}},
        "random_forest": {"model_type": "random_forest", "model_params": {}},
    }

    trained_models: dict[str, Any] = {}
    for model_name, config in configs.items():
        trained_models[model_name] = train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=config.get("model_type", "logistic_regression"),
            model_params=config.get("model_params", {}),
            sample_weight=sample_weight,
            random_state=random_state,
        )

    return trained_models
