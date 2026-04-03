"""Rule-based root-cause diagnosis for hiring bias analysis.

This module combines fairness metrics, dataset distribution analysis, and
feature correlation analysis to explain why bias is likely occurring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosisThresholds:
    """Thresholds used to diagnose likely root causes of bias."""

    historical_imbalance_alert: float = 0.50
    selection_rate_gap_moderate: float = 0.15
    selection_rate_gap_high: float = 0.25
    proxy_correlation_moderate: float = 0.50
    proxy_correlation_high: float = 0.70


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_imbalance_ratio(group_distribution: dict[str, Any] | None) -> float | None:
    """Return min/max representation ratio for the sensitive groups."""
    if not group_distribution:
        return None

    counts = [
        float(value)
        for value in group_distribution.values()
        if _to_float(value, -1.0) >= 0
    ]
    if len(counts) < 2:
        return None

    max_count = max(counts)
    if max_count == 0:
        return None

    return min(counts) / max_count


def _selection_rate_gap(selection_rate: dict[str, Any] | None) -> float | None:
    """Return the gap between max and min group selection rates."""
    if not selection_rate:
        return None

    rates = [float(value)
             for value in selection_rate.values() if value is not None]
    if len(rates) < 2:
        return None

    return max(rates) - min(rates)


def _proxy_features(
    feature_correlation_analysis: dict[str, Any] | None,
    threshold: float,
) -> list[tuple[str, float]]:
    """Return correlated features whose absolute correlation exceeds threshold."""
    if not feature_correlation_analysis:
        return []

    top_features = feature_correlation_analysis.get(
        "top_correlated_features", [])
    if not isinstance(top_features, list):
        return []

    flagged: list[tuple[str, float]] = []
    for item in top_features:
        if not isinstance(item, dict):
            continue
        feature = item.get("feature")
        correlation = _to_float(item.get("correlation"), 0.0)
        if feature is None:
            continue
        if abs(correlation) >= threshold:
            flagged.append((str(feature), abs(correlation)))

    return flagged


def _severity_from_signals(signal_strengths: list[float]) -> str:
    """Convert combined signal strength into a human-readable severity label."""
    if not signal_strengths:
        return "Low"

    strongest = max(signal_strengths)
    average = sum(signal_strengths) / len(signal_strengths)
    combined = 0.7 * strongest + 0.3 * average

    if combined >= 0.67:
        return "High"
    if combined >= 0.34:
        return "Medium"
    return "Low"


def diagnose_bias_root_causes(
    fairness_metrics: dict[str, Any],
    dataset_distribution_analysis: dict[str, Any],
    feature_correlation_analysis: dict[str, Any],
    thresholds: DiagnosisThresholds | None = None,
) -> dict[str, Any]:
    """Diagnose likely bias type and root causes using rule-based logic.

    Args:
        fairness_metrics: Output from fairness metric computation.
        dataset_distribution_analysis: Output from group distribution analysis.
        feature_correlation_analysis: Output from feature correlation analysis.
        thresholds: Optional threshold configuration.

    Returns:
        Dictionary with keys:
        - bias_type
        - root_causes
        - severity
    """
    if not isinstance(fairness_metrics, dict) or not fairness_metrics:
        raise ValueError("fairness_metrics must be a non-empty dictionary.")
    if not isinstance(dataset_distribution_analysis, dict) or not dataset_distribution_analysis:
        raise ValueError(
            "dataset_distribution_analysis must be a non-empty dictionary."
        )
    if not isinstance(feature_correlation_analysis, dict):
        raise ValueError("feature_correlation_analysis must be a dictionary.")

    cfg = thresholds or DiagnosisThresholds()

    group_distribution = dataset_distribution_analysis.get(
        "group_distribution")
    selection_rate = dataset_distribution_analysis.get("selection_rate")

    if group_distribution is not None and not isinstance(group_distribution, dict):
        raise ValueError(
            "group_distribution must be a dictionary if provided.")
    if selection_rate is not None and not isinstance(selection_rate, dict):
        raise ValueError("selection_rate must be a dictionary if provided.")

    imbalance_ratio = _group_imbalance_ratio(group_distribution)
    selection_gap = _selection_rate_gap(selection_rate)
    proxy_features = _proxy_features(
        feature_correlation_analysis,
        cfg.proxy_correlation_moderate,
    )

    triggered: list[dict[str, Any]] = []
    signal_strengths: list[float] = []

    # Historical bias: sensitive group representation is imbalanced.
    if imbalance_ratio is not None and imbalance_ratio < cfg.historical_imbalance_alert:
        historical_strength = min(
            1.0,
            (cfg.historical_imbalance_alert - imbalance_ratio)
            / cfg.historical_imbalance_alert,
        )
        triggered.append(
            {
                "bias_type": "Historical Bias",
                "root_cause": "Group distribution is highly imbalanced across the sensitive attribute.",
                "strength": historical_strength,
            }
        )
        signal_strengths.append(historical_strength)

    # Selection bias: hiring outcomes differ materially by group.
    if selection_gap is not None and selection_gap >= cfg.selection_rate_gap_moderate:
        selection_strength = min(
            1.0,
            selection_gap / cfg.selection_rate_gap_high,
        )
        triggered.append(
            {
                "bias_type": "Selection Bias",
                "root_cause": "Unequal selection rates across sensitive groups.",
                "strength": selection_strength,
            }
        )
        signal_strengths.append(selection_strength)

    # Proxy bias: features behave like indirect proxies for the sensitive attribute.
    if proxy_features:
        strongest_feature, strongest_corr = proxy_features[0]
        proxy_strength = min(1.0, strongest_corr / cfg.proxy_correlation_high)
        triggered.append(
            {
                "bias_type": "Proxy Bias",
                "root_cause": f"High correlation of {strongest_feature} with the sensitive attribute.",
                "strength": proxy_strength,
            }
        )
        signal_strengths.append(proxy_strength)

    if not triggered:
        return {
            "bias_type": "No Significant Bias",
            "root_causes": [],
            "severity": "Low",
        }

    bias_priority = {
        "Historical Bias": 0,
        "Selection Bias": 1,
        "Proxy Bias": 2,
    }
    strongest = max(
        triggered,
        key=lambda item: (
            item["strength"],
            -bias_priority.get(item["bias_type"], 99),
        ),
    )

    root_causes = [item["root_cause"] for item in triggered]

    return {
        "bias_type": strongest["bias_type"],
        "root_causes": root_causes,
        "severity": _severity_from_signals(signal_strengths),
    }
