"""Higher-level SHAP analysis helpers used by reporting and dashboards."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.explainability.group_explainer import analyze_group_shap_values
from src.explainability.shap_explainer import compute_shap_values


def summarize_global_shap_importance(
        model: Any,
        X: pd.DataFrame,
        top_n: int = 10,
) -> pd.DataFrame:
    """Return top global SHAP importances as a tidy dataframe."""
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    result = compute_shap_values(
        model=model, X=X, feature_names=list(X.columns))
    shap_df = pd.DataFrame(result["shap_values"],
                           columns=result["feature_names"])
    importance = shap_df.abs().mean().sort_values(ascending=False)

    output = importance.head(top_n).rename("mean_abs_shap").reset_index()
    output.columns = ["feature", "mean_abs_shap"]
    return output


def summarize_group_shap_disparities(
        model: Any,
        X: pd.DataFrame,
        sensitive_features: pd.Series,
        top_n: int = 10,
) -> pd.DataFrame:
    """Return top features with largest SHAP spread across sensitive groups."""
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    group_values = analyze_group_shap_values(
        model=model,
        X=X,
        sensitive_features=sensitive_features,
        feature_names=list(X.columns),
    )
    comparison_df = pd.DataFrame(group_values).T
    if comparison_df.empty:
        return pd.DataFrame(columns=["feature", "group_spread"])

    spread = (comparison_df.max(axis=0) - comparison_df.min(axis=0)).sort_values(
        ascending=False
    )
    output = spread.head(top_n).rename("group_spread").reset_index()
    output.columns = ["feature", "group_spread"]
    return output
