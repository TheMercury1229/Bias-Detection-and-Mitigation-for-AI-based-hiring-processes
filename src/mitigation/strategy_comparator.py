"""Comparison helpers for simulated mitigation strategies.

This module ranks mitigation strategies based on fairness improvement and
accuracy retention so users can choose the best trade-off.
"""

from __future__ import annotations

from typing import Any


def _validate_results(simulation_results: list[dict[str, Any]]) -> None:
    if not isinstance(simulation_results, list):
        raise ValueError("simulation_results must be a list.")


def _empty_report() -> dict[str, Any]:
    return {
        "sorted_results": [],
        "best_fairness_strategy": {
            "strategy": None,
            "fairness": None,
            "accuracy": None,
        },
        "best_accuracy_strategy": {
            "strategy": None,
            "accuracy": None,
            "fairness": None,
        },
        "balanced_choice": {
            "strategy": None,
            "rank_score": None,
            "fairness": None,
            "accuracy": None,
        },
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compare_mitigation_strategies(
    simulation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank mitigation strategies based on fairness vs accuracy trade-offs."""
    _validate_results(simulation_results)
    if not simulation_results:
        return _empty_report()

    normalized_results: list[dict[str, Any]] = []
    best_gain = max(
        max(_as_float(result.get("fairness_improvement"), 0.0), 0.0)
        for result in simulation_results
    ) or 1.0

    for result in simulation_results:
        fairness_improvement = _as_float(
            result.get("fairness_improvement"), 0.0)
        accuracy_retention = _as_float(result.get("accuracy_retention"), 0.0)
        normalized_gain = max(fairness_improvement, 0.0) / \
            best_gain if best_gain else 0.0
        rank_score = round((0.65 * normalized_gain) +
                           (0.35 * accuracy_retention), 6)

        enriched = dict(result)
        enriched["normalized_fairness_improvement"] = float(
            round(normalized_gain, 6)
        )
        enriched["rank_score"] = rank_score
        normalized_results.append(enriched)

    sorted_results = sorted(
        normalized_results,
        key=lambda item: (
            item["rank_score"],
            item.get("fairness_improvement", 0.0),
            item.get("accuracy", 0.0),
        ),
        reverse=True,
    )

    best_fairness_strategy = min(
        normalized_results,
        key=lambda item: (
            _as_float(item.get("fairness"), 1.0),
            -_as_float(item.get("accuracy"), 0.0),
        ),
    )
    best_accuracy_strategy = max(
        normalized_results,
        key=lambda item: (
            _as_float(item.get("accuracy"), 0.0),
            _as_float(item.get("fairness_improvement"), 0.0),
        ),
    )
    balanced_choice = sorted_results[0]

    return {
        "sorted_results": sorted_results,
        "best_fairness_strategy": {
            "strategy": best_fairness_strategy["strategy"],
            "fairness": float(best_fairness_strategy["fairness"]),
            "accuracy": float(best_fairness_strategy["accuracy"]),
        },
        "best_accuracy_strategy": {
            "strategy": best_accuracy_strategy["strategy"],
            "accuracy": float(best_accuracy_strategy["accuracy"]),
            "fairness": float(best_accuracy_strategy["fairness"]),
        },
        "balanced_choice": {
            "strategy": balanced_choice["strategy"],
            "rank_score": float(balanced_choice["rank_score"]),
            "fairness": float(balanced_choice["fairness"]),
            "accuracy": float(balanced_choice["accuracy"]),
        },
    }
