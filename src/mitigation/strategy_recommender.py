"""Mitigation strategy recommendations for fairness-aware hiring ML systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyRecommendation:
    """Represents a mitigation strategy recommendation."""

    name: str
    reason: str


DEFAULT_STRATEGY_MAP: dict[str, list[StrategyRecommendation]] = {
    "Selection Bias": [
        StrategyRecommendation(
            name="Fairness-Constrained Learning",
            reason=(
                "Apply in-processing constraints (for example demographic parity or "
                "equalized-odds constraints) to reduce selection-rate disparities."
            ),
        ),
        StrategyRecommendation(
            name="Reweighing",
            reason=(
                "Adjust sample weights by sensitive group and label to reduce "
                "imbalanced selection patterns during training."
            ),
        ),
    ],
    "Prediction Bias": [
        StrategyRecommendation(
            name="Post-Processing Threshold Optimization",
            reason=(
                "Tune decision thresholds per group to reduce unequal false positive "
                "and true positive rates after model training."
            ),
        ),
        StrategyRecommendation(
            name="Equalized Odds Post-Processing",
            reason=(
                "Use equalized-odds post-processing methods to align error rates "
                "across demographic groups."
            ),
        ),
    ],
    "Historical Bias": [
        StrategyRecommendation(
            name="Data Reweighing",
            reason=(
                "Reweight underrepresented group examples to reduce the impact of "
                "historical imbalance in training data."
            ),
        ),
        StrategyRecommendation(
            name="Resampling",
            reason=(
                "Use over/under-sampling to improve representation balance before "
                "model training."
            ),
        ),
    ],
    "Proxy Bias": [
        StrategyRecommendation(
            name="Feature Debiasing",
            reason=(
                "Transform or regularize features correlated with sensitive "
                "attributes to reduce indirect discrimination."
            ),
        ),
        StrategyRecommendation(
            name="Feature Removal or Auditing",
            reason=(
                "Identify and remove proxy variables that leak sensitive information "
                "into model decisions."
            ),
        ),
    ],
    "No Significant Bias": [
        StrategyRecommendation(
            name="Continuous Fairness Monitoring",
            reason=(
                "No major bias signal detected now, but metrics should be monitored "
                "across retraining cycles and new hiring cohorts."
            ),
        )
    ],
}


def _normalize_bias_type(bias_type: str) -> str:
    """Normalize bias labels to supported categories."""
    normalized = (bias_type or "").strip().lower()

    alias_map = {
        "selection bias": "Selection Bias",
        "prediction bias": "Prediction Bias",
        "historical bias": "Historical Bias",
        "proxy bias": "Proxy Bias",
        "no significant bias": "No Significant Bias",
    }

    if normalized in alias_map:
        return alias_map[normalized]

    # Accept exact canonical labels if user passes them directly.
    if bias_type in DEFAULT_STRATEGY_MAP:
        return bias_type

    return "No Significant Bias"


def recommend_mitigation_strategies(
        bias_type: str,
        strategy_map: dict[str, list[StrategyRecommendation]] | None = None,
) -> dict[str, Any]:
    """Return mitigation recommendations for the detected bias type.

    Args:
            bias_type: Detected bias label (for example, Selection Bias).
            strategy_map: Optional custom strategy mapping for extension.

    Returns:
            Structured dictionary containing selected bias type and recommendations.
    """
    if not isinstance(bias_type, str) or not bias_type.strip():
        raise ValueError("bias_type must be a non-empty string.")

    mapping = strategy_map or DEFAULT_STRATEGY_MAP
    canonical_bias_type = _normalize_bias_type(bias_type)
    recommendations = mapping.get(
        canonical_bias_type, mapping["No Significant Bias"])

    return {
        "bias_type": canonical_bias_type,
        "recommended_strategies": [asdict(strategy) for strategy in recommendations],
    }
