"""Rule-based bias identification utilities for hiring fairness analysis.
This module interprets fairness metrics and classifies likely bias type(s)
using transparent threshold rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BiasRuleThresholds:
    """Thresholds used by the rule-based classifier.

    Notes:
    - disparate_impact_ratio uses the common 80% rule heuristic (0.8).
    - equalized_odds_difference and demographic_parity_difference are interpreted
      using practical cutoffs for low/moderate/high concern.
    - historical imbalance ratio is min_group_share / max_group_share.
    """

    disparate_impact_alert: float = 0.80
    equalized_odds_moderate: float = 0.10
    equalized_odds_high: float = 0.20
    demographic_parity_moderate: float = 0.10
    demographic_parity_high: float = 0.20
    historical_imbalance_alert: float = 0.50


def _to_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Metric '{key}' must be numeric, got: {value!r}") from None


def _get_distribution_ratio(distribution: dict[str, float] | None) -> float | None:
    """Return min/max group representation ratio for imbalance checks.

    Input may be raw group counts or normalized shares.
    """
    if not distribution:
        return None

    values = [float(v) for v in distribution.values() if float(v) >= 0]
    if len(values) < 2:
        return None

    max_val = max(values)
    min_val = min(values)
    if max_val == 0:
        return None

    return min_val / max_val


def _severity_from_signals(signal_strengths: list[float]) -> float:
    """Aggregate signal strengths into a normalized severity score [0, 1]."""
    if not signal_strengths:
        return 0.0

    max_signal = max(signal_strengths)
    avg_signal = sum(signal_strengths) / len(signal_strengths)
    severity = 0.7 * max_signal + 0.3 * avg_signal
    return float(round(min(1.0, max(0.0, severity)), 3))


def identify_bias_type(
        fairness_metrics: dict[str, Any],
        thresholds: BiasRuleThresholds | None = None,
) -> dict[str, Any]:
    """Classify likely bias type from fairness metrics using rule-based logic.

    Expected keys in fairness_metrics:
    - demographic_parity_difference: float
    - equalized_odds_difference: float
    - disparate_impact_ratio: float
    - selection_rate_by_group: dict[str, float]

    Optional keys for stronger diagnosis:
    - sensitive_group_distribution: dict[str, float]
      (counts or shares for each sensitive group in the dataset)

    Returns:
            Dictionary containing:
            - detected_bias_type: str
            - explanation: str
            - severity_score: float in [0, 1]
            - triggered_rules: list[str]
            - supporting_metrics: dict[str, float | None]
    """
    if not fairness_metrics:
        raise ValueError("fairness_metrics must be a non-empty dictionary.")

    cfg = thresholds or BiasRuleThresholds()

    dp_diff = abs(_to_float(fairness_metrics,
                  "demographic_parity_difference", 0.0))
    eo_diff = abs(_to_float(fairness_metrics,
                  "equalized_odds_difference", 0.0))
    di_ratio = _to_float(fairness_metrics, "disparate_impact_ratio", 1.0)
    selection_rate_by_group = fairness_metrics.get(
        "selection_rate_by_group", {})
    sensitive_distribution = fairness_metrics.get(
        "sensitive_group_distribution")

    if not isinstance(selection_rate_by_group, dict):
        raise ValueError("selection_rate_by_group must be a dictionary.")

    imbalance_ratio = None
    if sensitive_distribution is not None:
        if not isinstance(sensitive_distribution, dict):
            raise ValueError(
                "sensitive_group_distribution must be a dictionary.")
        imbalance_ratio = _get_distribution_ratio(sensitive_distribution)

    triggered_rules: list[str] = []
    signal_strengths: list[float] = []

    # Selection bias signal: 80% rule style threshold and parity gap.
    if di_ratio < cfg.disparate_impact_alert:
        triggered_rules.append(
            "Selection Bias signal: disparate impact ratio below 0.80 heuristic."
        )
        signal_strengths.append(
            min(1.0, (cfg.disparate_impact_alert - di_ratio) / 0.40))

    if dp_diff >= cfg.demographic_parity_moderate:
        triggered_rules.append(
            "Selection Bias signal: notable demographic parity difference across groups."
        )
        signal_strengths.append(min(1.0, dp_diff / 0.50))

    # Prediction bias signal: unequal TPR/FPR behavior across groups.
    if eo_diff >= cfg.equalized_odds_moderate:
        triggered_rules.append(
            "Prediction Bias signal: equalized odds difference indicates performance disparity."
        )
        signal_strengths.append(min(1.0, eo_diff / 0.50))

    # Historical bias signal: group representation heavily imbalanced.
    if imbalance_ratio is not None and imbalance_ratio < cfg.historical_imbalance_alert:
        triggered_rules.append(
            "Historical Bias signal: sensitive-group representation is highly imbalanced."
        )
        signal_strengths.append(
            min(1.0, (cfg.historical_imbalance_alert - imbalance_ratio) / 0.50))

    # Proxy bias signal: allocation disparity with relatively smaller error-rate disparity.
    if (
            dp_diff >= cfg.demographic_parity_high
            and eo_diff < cfg.equalized_odds_moderate
            and di_ratio < 1.0
    ):
        triggered_rules.append(
            "Proxy Bias signal: outcome disparity is high while equalized-odds disparity is lower."
        )
        signal_strengths.append(min(1.0, dp_diff / 0.60))

    category_scores = {
        "Selection Bias": 0.0,
        "Prediction Bias": 0.0,
        "Historical Bias": 0.0,
        "Proxy Bias": 0.0,
    }

    if di_ratio < cfg.disparate_impact_alert or dp_diff >= cfg.demographic_parity_moderate:
        category_scores["Selection Bias"] = max(
            category_scores["Selection Bias"],
            max(0.0, min(1.0, (cfg.disparate_impact_alert - di_ratio) / 0.40)),
            max(0.0, min(1.0, dp_diff / 0.50)),
        )

    if eo_diff >= cfg.equalized_odds_moderate:
        category_scores["Prediction Bias"] = max(
            category_scores["Prediction Bias"],
            max(0.0, min(1.0, eo_diff / 0.50)),
        )

    if imbalance_ratio is not None and imbalance_ratio < cfg.historical_imbalance_alert:
        category_scores["Historical Bias"] = max(
            category_scores["Historical Bias"],
            max(0.0, min(1.0, (cfg.historical_imbalance_alert - imbalance_ratio) / 0.50)),
        )

    if (
            dp_diff >= cfg.demographic_parity_high
            and eo_diff < cfg.equalized_odds_moderate
            and di_ratio < 1.0
    ):
        category_scores["Proxy Bias"] = max(
            category_scores["Proxy Bias"],
            max(0.0, min(1.0, dp_diff / 0.60)),
        )

    best_bias_type = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_bias_type]

    if best_score < 0.20 and not triggered_rules:
        detected_bias_type = "No Significant Bias"
        explanation = (
            "Fairness metrics do not cross the configured rule thresholds for "
            "selection, prediction, historical, or proxy bias."
        )
    else:
        detected_bias_type = best_bias_type
        explanation_parts = [
            f"Detected {detected_bias_type} based on threshold rules.",
            f"Demographic parity difference={dp_diff:.3f}",
            f"Equalized odds difference={eo_diff:.3f}",
            f"Disparate impact ratio={di_ratio:.3f}",
        ]
        if imbalance_ratio is not None:
            explanation_parts.append(
                f"Sensitive distribution imbalance ratio={imbalance_ratio:.3f}")
        if triggered_rules:
            explanation_parts.append(
                "Triggered rules: " + " | ".join(triggered_rules))
        explanation = "; ".join(explanation_parts)

    severity_score = _severity_from_signals(signal_strengths)

    return {
        "detected_bias_type": detected_bias_type,
        "explanation": explanation,
        "severity_score": severity_score,
        "triggered_rules": triggered_rules,
        "supporting_metrics": {
            "demographic_parity_difference": round(dp_diff, 6),
            "equalized_odds_difference": round(eo_diff, 6),
            "disparate_impact_ratio": round(di_ratio, 6),
            "distribution_imbalance_ratio": (
                round(imbalance_ratio,
                      6) if imbalance_ratio is not None else None
            ),
        },
    }
