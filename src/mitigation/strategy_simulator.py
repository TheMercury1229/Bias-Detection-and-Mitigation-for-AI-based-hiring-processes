"""Strategy simulation utilities for fairness-aware mitigation planning.

This module evaluates mitigation strategies temporarily so users can compare
fairness and accuracy trade-offs before applying a strategy permanently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.mitigation.mitigation_methods import (
    apply_data_reweighing,
    train_fairness_constrained_model,
    train_threshold_optimizer,
)
from src.models.evaluate_model import evaluate_predictions
from src.models.train_model import build_model, train_baseline_model, train_model


def _to_dataframe(values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True)
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Feature matrix must be 2D.")
    return pd.DataFrame(array)


def _to_series(values: pd.Series | np.ndarray | list[Any], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(np.asarray(values).ravel(), name=name)


def _normalize_strategy_name(strategy: str) -> str:
    normalized = (strategy or "").strip().lower()
    alias_map = {
        "reweighing": "Reweighing",
        "data reweighing": "Reweighing",
        "fairness constrained learning": "Fairness-Constrained Learning",
        "fairness constraint": "Fairness-Constrained Learning",
        "post-processing threshold optimization": "Post-Processing Threshold Optimization",
        "equalized odds post-processing": "Equalized Odds Post-Processing",
        "feature debiasing": "Feature Debiasing",
        "feature removal or auditing": "Feature Removal or Auditing",
    }
    return alias_map.get(normalized, strategy.strip())


def _infer_model_type(model: Any | None, fallback: str = "logistic_regression") -> str:
    if model is None:
        return fallback

    model_name = type(model).__name__.lower()
    if "randomforest" in model_name:
        return "random_forest"
    if "logistic" in model_name:
        return "logistic_regression"
    return fallback


def _fairness_score(fairness_metrics: dict[str, Any]) -> float:
    dp_diff = abs(float(fairness_metrics.get(
        "demographic_parity_difference", 0.0)))
    eo_diff = abs(float(fairness_metrics.get(
        "equalized_odds_difference", 0.0)))
    di_ratio = float(fairness_metrics.get("disparate_impact_ratio", 1.0))
    di_gap = abs(1.0 - di_ratio)

    score = (min(dp_diff, 1.0) + min(eo_diff, 1.0) + min(di_gap, 1.0)) / 3.0
    return float(round(score, 6))


def _clone_or_build_estimator(model: Any | None, model_type: str, random_state: int) -> Any:
    if model is not None and hasattr(model, "get_params"):
        try:
            return clone(model)
        except Exception:
            pass
    return build_model(model_type=model_type, random_state=random_state)


def _simulate_strategy(
    strategy: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_train: pd.Series,
    sensitive_test: pd.Series,
    *,
    sensitive_column: str | None,
    model: Any | None,
    model_type: str,
    random_state: int,
) -> dict[str, Any]:
    normalized_strategy = _normalize_strategy_name(strategy)
    base_estimator = _clone_or_build_estimator(model, model_type, random_state)

    if normalized_strategy in {"Reweighing"}:
        rw = apply_data_reweighing(X_train, y_train, sensitive_train)
        temp_model = train_model(
            X_train=rw.X_train,
            y_train=rw.y_train,
            model_type=model_type,
            sample_weight=rw.sample_weights,
            random_state=random_state,
        )
        y_pred = temp_model.predict(X_test)

    elif normalized_strategy == "Fairness-Constrained Learning":
        temp_model = train_fairness_constrained_model(
            X_train=X_train,
            y_train=y_train,
            sensitive_features=sensitive_train,
            constraint="demographic_parity",
            base_estimator=base_estimator,
            random_state=random_state,
        )
        y_pred = temp_model.predict(X_test)

    elif normalized_strategy in {
        "Post-Processing Threshold Optimization",
        "Equalized Odds Post-Processing",
    }:
        constraints = (
            "equalized_odds"
            if normalized_strategy == "Equalized Odds Post-Processing"
            else "demographic_parity"
        )
        temp_model = train_threshold_optimizer(
            estimator=base_estimator,
            X_train=X_train,
            y_train=y_train,
            sensitive_features=sensitive_train,
            constraints=constraints,
            predict_method="predict_proba",
        )
        y_pred = temp_model.predict(X_test, sensitive_features=sensitive_test)

    elif normalized_strategy in {"Feature Debiasing", "Feature Removal or Auditing"}:
        cols_to_drop = [
            col
            for col in X_train.columns
            if (
                (sensitive_column is not None and col == sensitive_column)
                or (sensitive_column is not None and col.startswith(f"{sensitive_column}_"))
                or col.startswith("gender_")
                or col.startswith("sex_")
                or col.startswith("race_")
                or col.startswith("age_")
                or col.startswith("ethnicity_")
            )
        ]
        X_train_debiased = X_train.drop(columns=cols_to_drop, errors="ignore")
        X_test_debiased = X_test.drop(columns=cols_to_drop, errors="ignore")
        temp_model = train_baseline_model(
            X_train=X_train_debiased,
            y_train=y_train,
            model_type=model_type,
            random_state=random_state,
        )
        y_pred = temp_model.predict(X_test_debiased)

    else:
        temp_model = model if model is not None else train_baseline_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            random_state=random_state,
        )
        y_pred = temp_model.predict(X_test)

    evaluation = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
        model_name=normalized_strategy,
    )

    performance = evaluation["performance"]
    fairness_metrics = evaluation["fairness"]
    fairness_score = _fairness_score(fairness_metrics)

    return {
        "strategy": normalized_strategy,
        "accuracy": float(performance["accuracy"]),
        "fairness": fairness_score,
        "performance": performance,
        "fairness_metrics": fairness_metrics,
    }


def simulate_mitigation_strategies(
    *,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray | list[Any],
    y_test: pd.Series | np.ndarray | list[Any],
    sensitive_train: pd.Series | np.ndarray | list[Any],
    sensitive_test: pd.Series | np.ndarray | list[Any],
    strategies: list[str],
    model: Any | None = None,
    model_type: str | None = None,
    sensitive_column: str | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate mitigation strategies temporarily without modifying the dataset."""
    if not strategies:
        raise ValueError("strategies must be a non-empty list.")

    X_train_df = _to_dataframe(X_train)
    X_test_df = _to_dataframe(X_test)
    y_train_series = _to_series(y_train, name="target")
    y_test_series = _to_series(y_test, name="target")
    sensitive_train_series = _to_series(
        sensitive_train, name="sensitive_train")
    sensitive_test_series = _to_series(sensitive_test, name="sensitive_test")

    if not (
        len(X_train_df) == len(y_train_series) == len(sensitive_train_series)
        and len(X_test_df) == len(y_test_series) == len(sensitive_test_series)
    ):
        raise ValueError("Input lengths must align for train/test splits.")

    inferred_model_type = model_type or _infer_model_type(model)

    if model is None and inferred_model_type == "logistic_regression":
        baseline_model = train_baseline_model(
            X_train=X_train_df,
            y_train=y_train_series,
            model_type=inferred_model_type,
            random_state=random_state,
        )
    else:
        baseline_model = model

    baseline_predictions = baseline_model.predict(X_test_df)
    baseline_evaluation = evaluate_predictions(
        y_true=y_test_series,
        y_pred=baseline_predictions,
        sensitive_features=sensitive_test_series,
        model_name="baseline",
    )
    baseline_accuracy = float(baseline_evaluation["performance"]["accuracy"])
    baseline_fairness = _fairness_score(baseline_evaluation["fairness"])

    simulation_results: list[dict[str, Any]] = []
    seen_strategies: set[str] = set()
    for strategy in strategies:
        normalized_strategy = _normalize_strategy_name(strategy)
        if normalized_strategy in seen_strategies:
            continue
        seen_strategies.add(normalized_strategy)

        result = _simulate_strategy(
            strategy=normalized_strategy,
            X_train=X_train_df,
            X_test=X_test_df,
            y_train=y_train_series,
            y_test=y_test_series,
            sensitive_train=sensitive_train_series,
            sensitive_test=sensitive_test_series,
            sensitive_column=sensitive_column,
            model=baseline_model,
            model_type=inferred_model_type,
            random_state=random_state,
        )
        result["baseline_accuracy"] = baseline_accuracy
        result["baseline_fairness"] = baseline_fairness
        result["fairness_improvement"] = float(
            round(baseline_fairness - result["fairness"], 6)
        )
        result["accuracy_retention"] = float(
            round(result["accuracy"] / baseline_accuracy, 6)
            if baseline_accuracy > 0
            else 0.0
        )
        simulation_results.append(result)

    return simulation_results
