"""Human-readable bias explanations from SHAP group comparisons.

This module turns group-level SHAP differences into plain-language insights
that can be consumed by non-technical users such as HR teams.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any


def _validate_group_shap_values(group_shap_values: dict[str, Any]) -> None:
    """Validate the nested group-to-feature SHAP structure."""
    if not isinstance(group_shap_values, dict) or not group_shap_values:
        raise ValueError("group_shap_values must be a non-empty dictionary.")

    for group_name, feature_values in group_shap_values.items():
        if not isinstance(feature_values, dict) or not feature_values:
            raise ValueError(
                f"Feature values for group '{group_name}' must be a non-empty dictionary."
            )


def _format_feature_name(feature_name: str) -> str:
    """Convert feature names into a friendlier display label."""
    return feature_name.replace("_", " ").strip()


def _build_pairwise_insight(
    feature_name: str,
    group_a: str,
    value_a: float,
    group_b: str,
    value_b: float,
) -> str:
    """Create a readable explanation for a feature difference between groups."""
    readable_feature = _format_feature_name(feature_name)

    if value_a < 0 and value_b < 0:
        stronger_group = group_a if abs(value_a) > abs(value_b) else group_b
        weaker_group = group_b if stronger_group == group_a else group_a
        return (
            f"{readable_feature} has a higher negative impact on {stronger_group} "
            f"candidates than on {weaker_group} candidates."
        )

    if value_a >= 0 and value_b >= 0:
        stronger_group = group_a if abs(value_a) > abs(value_b) else group_b
        weaker_group = group_b if stronger_group == group_a else group_a
        return (
            f"{readable_feature} influences {stronger_group} candidates more strongly "
            f"than {weaker_group} candidates."
        )

    if value_a < value_b:
        return (
            f"{readable_feature} has a more negative effect on {group_a} candidates "
            f"than on {group_b} candidates."
        )

    return (
        f"{readable_feature} has a more negative effect on {group_b} candidates "
        f"than on {group_a} candidates."
    )


def generate_bias_explanations(
    group_shap_values: dict[str, dict[str, float]],
    top_n: int = 5,
    difference_threshold: float = 0.05,
) -> dict[str, list[str]]:
    """Generate plain-language bias insights from group SHAP values.

    Args:
        group_shap_values: Output from group-level SHAP comparison.
        top_n: Maximum number of insights to return.
        difference_threshold: Minimum mean absolute difference needed to flag a feature.

    Returns:
        Dictionary with an "insights" list containing human-readable statements.
        Example:
        {
            "insights": [
                "employment_gap has higher negative impact on female candidates",
                "university_rank influences male candidates more strongly"
            ]
        }
    """
    _validate_group_shap_values(group_shap_values)

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    if difference_threshold < 0:
        raise ValueError("difference_threshold must be non-negative.")

    feature_names = sorted(
        {feature for group_values in group_shap_values.values()
         for feature in group_values}
    )

    ranked_differences: list[tuple[float, str]] = []
    group_names = list(group_shap_values.keys())

    for feature_name in feature_names:
        values_by_group = {
            group_name: float(group_values[feature_name])
            for group_name, group_values in group_shap_values.items()
            if feature_name in group_values
        }

        if len(values_by_group) < 2:
            continue

        for group_a, group_b in combinations(values_by_group.keys(), 2):
            value_a = values_by_group[group_a]
            value_b = values_by_group[group_b]
            difference = abs(value_a - value_b)

            if difference < difference_threshold:
                continue

            insight = _build_pairwise_insight(
                feature_name=feature_name,
                group_a=group_a,
                value_a=value_a,
                group_b=group_b,
                value_b=value_b,
            )
            ranked_differences.append((difference, insight))

    ranked_differences.sort(key=lambda item: item[0], reverse=True)

    insights: list[str] = []
    seen_insights: set[str] = set()
    for _, insight in ranked_differences:
        normalized = insight.lower()
        if normalized in seen_insights:
            continue
        seen_insights.add(normalized)
        insights.append(insight)
        if len(insights) >= top_n:
            break

    if not insights:
        insights = [
            "No large SHAP differences were found across demographic groups."
        ]

    return {"insights": insights}
