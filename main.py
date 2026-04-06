"""Main pipeline for hiring fairness analysis and mitigation.

Example:
python main.py --data data/hiring.csv --target hired --sensitive gender
"""

from __future__ import annotations
from src.models.train_model import build_model, train_baseline_model, train_model
from src.models.evaluate_model import compare_baseline_and_mitigated_models, evaluate_predictions
from src.mitigation.strategy_simulator import simulate_mitigation_strategies
from src.mitigation.strategy_recommender import recommend_mitigation_strategies
from src.mitigation.strategy_comparator import compare_mitigation_strategies
from src.mitigation.mitigation_methods import (
    apply_data_reweighing,
    train_fairness_constrained_model,
    train_threshold_optimizer,
)
from src.data.schema_validator import validate_dataset_schema
from src.data.preprocess import preprocess_dataset
from src.data.load_data import load_dataset
from src.bias_detection.data_analyzer import (
    analyze_group_distribution_and_selection_rate,
)
from src.bias_detection.correlation_analyzer import (
    analyze_feature_correlation_with_sensitive_attribute,
)
from src.bias_detection.bias_diagnosis import diagnose_bias_root_causes
from src.bias_detection.bias_identifier import identify_bias_type
from joblib import dump, load
import pandas as pd

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Configure runtime before importing modules that may transitively load TensorFlow/inFairness.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"inFairness.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"keras.*")


def ensure_output_dirs(base_dir: str | Path = "results") -> dict[str, Path]:
    """Create and return project output directories."""
    base = Path(base_dir)
    reports_dir = base / "reports"
    plots_dir = base / "plots"
    models_dir = base / "models"

    base.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "reports": reports_dir,
        "plots": plots_dir,
        "models": models_dir,
    }


def save_model_artifacts(
        baseline_model: Any,
        mitigated_model: Any,
        selected_strategy: str,
        target_column: str,
        sensitive_column: str,
        model_type: str,
        output_paths: dict[str, Path],
) -> dict[str, str]:
    """Persist trained models and metadata to the results/models directory."""
    models_dir = output_paths["models"]

    baseline_model_path = models_dir / "baseline_model.joblib"
    mitigated_model_path = models_dir / "mitigated_model.joblib"
    metadata_path = models_dir / "model_metadata.json"

    dump(baseline_model, baseline_model_path)
    dump(mitigated_model, mitigated_model_path)

    metadata = {
        "baseline_model_file": baseline_model_path.name,
        "mitigated_model_file": mitigated_model_path.name,
        "selected_mitigation_strategy": selected_strategy,
        "target_column": target_column,
        "sensitive_column": sensitive_column,
        "baseline_model_type": model_type,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "baseline_model": str(baseline_model_path),
        "mitigated_model": str(mitigated_model_path),
        "metadata": str(metadata_path),
    }


def load_saved_model(model_path: str | Path) -> Any:
    """Load a persisted model artifact for inference or evaluation."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return load(path)


def apply_mitigation_strategy(
        strategy_name: str,
        baseline_state: dict[str, Any],
) -> tuple[Any, Any]:
    """Apply selected mitigation strategy and return model plus predictions."""
    X_train = baseline_state["X_train"]
    X_test = baseline_state["X_test"]
    y_train = baseline_state["y_train"]
    sensitive_train = baseline_state["sensitive_train"]
    sensitive_test = baseline_state["sensitive_test"]
    sensitive_column = baseline_state["sensitive_column"]

    if strategy_name in {"Reweighing", "Data Reweighing"}:
        rw = apply_data_reweighing(X_train, y_train, sensitive_train)
        model = train_model(
            X_train=rw.X_train,
            y_train=rw.y_train,
            model_type="logistic_regression",
            sample_weight=rw.sample_weights,
            random_state=42,
        )
        y_pred = model.predict(X_test)
        return model, y_pred

    if strategy_name == "Fairness-Constrained Learning":
        model = train_fairness_constrained_model(
            X_train=X_train,
            y_train=y_train,
            sensitive_features=sensitive_train,
            constraint="demographic_parity",
        )
        y_pred = model.predict(X_test)
        return model, y_pred

    if strategy_name in {
            "Post-Processing Threshold Optimization",
            "Equalized Odds Post-Processing",
    }:
        constraints = (
            "equalized_odds"
            if strategy_name == "Equalized Odds Post-Processing"
            else "demographic_parity"
        )
        threshold_model = train_threshold_optimizer(
            estimator=build_model("logistic_regression", random_state=42),
            X_train=X_train,
            y_train=y_train,
            sensitive_features=sensitive_train,
            constraints=constraints,
            predict_method="predict_proba",
        )
        y_pred = threshold_model.predict(
            X_test, sensitive_features=sensitive_test)
        return threshold_model, y_pred

    if strategy_name in {"Feature Debiasing", "Feature Removal or Auditing"}:
        cols_to_drop = [
            col
            for col in X_train.columns
            if col == sensitive_column or col.startswith(f"{sensitive_column}_")
        ]
        X_train_debiased = X_train.drop(columns=cols_to_drop, errors="ignore")
        X_test_debiased = X_test.drop(columns=cols_to_drop, errors="ignore")
        model = train_baseline_model(
            X_train=X_train_debiased,
            y_train=y_train,
            model_type="logistic_regression",
            random_state=42,
        )
        y_pred = model.predict(X_test_debiased)
        return model, y_pred

    model = baseline_state["baseline_model"]
    y_pred = baseline_state["y_pred_baseline"]
    return model, y_pred


def _print_strategy_simulation_report(comparison_report: dict[str, Any]) -> None:
    """Print a readable simulation summary for CLI users."""
    simulation_df = pd.DataFrame(comparison_report["sorted_results"])
    if simulation_df.empty:
        print("No strategy simulation results available.")
        return

    display_columns = [
        "strategy",
        "accuracy",
        "fairness",
        "fairness_improvement",
        "accuracy_retention",
        "rank_score",
    ]
    available_columns = [
        col for col in display_columns if col in simulation_df.columns]
    print("\nStrategy simulation results:")
    print(simulation_df[available_columns].to_string(index=False))
    print(
        "Best fairness strategy: "
        f"{comparison_report['best_fairness_strategy']['strategy']}"
    )
    print(
        "Best accuracy strategy: "
        f"{comparison_report['best_accuracy_strategy']['strategy']}"
    )
    print(
        "Balanced choice: "
        f"{comparison_report['balanced_choice']['strategy']}"
    )


def _select_strategy_from_comparison(
        comparison_report: dict[str, Any],
        preferred_strategy: str | None = None,
        interactive: bool = False,
) -> str:
    """Choose a mitigation strategy from ranked simulation results."""
    ranked_results = comparison_report.get("sorted_results", [])
    if not ranked_results:
        return "Continuous Fairness Monitoring"

    available_strategies = [result["strategy"] for result in ranked_results]
    balanced_choice = comparison_report["balanced_choice"]["strategy"]

    if preferred_strategy:
        if preferred_strategy not in available_strategies:
            raise ValueError(
                f"Selected strategy '{preferred_strategy}' was not part of the simulation results."
            )
        return preferred_strategy

    if not interactive:
        return balanced_choice

    print("\nAvailable mitigation strategies:")
    for index, result in enumerate(ranked_results, start=1):
        print(
            f"{index}. {result['strategy']} | accuracy={result['accuracy']:.3f} | "
            f"fairness={result['fairness']:.3f} | rank={result['rank_score']:.3f}"
        )

    prompt = (
        f"Choose a strategy by number or name [default: {balanced_choice}]: "
    )
    selection = input(prompt).strip()
    if not selection:
        return balanced_choice

    if selection.isdigit():
        index = int(selection)
        if 1 <= index <= len(ranked_results):
            return ranked_results[index - 1]["strategy"]
        raise ValueError("Strategy selection number is out of range.")

    if selection in available_strategies:
        return selection

    raise ValueError(
        f"Unknown strategy selection '{selection}'. Choose one of: {', '.join(available_strategies)}"
    )


def run_pipeline(
        data_path: str,
        target_column: str,
        sensitive_column: str,
        model_type: str,
        test_size: float,
        output_dir: str,
    preferred_strategy: str | None = None,
    interactive_selection: bool = False,
) -> dict[str, Any]:
    """Run the full baseline vs mitigated fairness pipeline."""
    output_paths = ensure_output_dirs(output_dir)

    df = load_dataset(data_path, verbose=True)
    validate_dataset_schema(
        df=df,
        required_columns=[col for col in df.columns if col != target_column],
        target_column=target_column,
        sensitive_attributes=[sensitive_column],
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_train,
        sensitive_test,
    ) = preprocess_dataset(
        df=df,
        target_column=target_column,
        sensitive_attributes=sensitive_column,
        test_size=test_size,
        random_state=42,
        include_sensitive_in_features=True,
    )

    baseline_model = train_baseline_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        random_state=42,
    )
    y_pred_baseline = baseline_model.predict(X_test)

    baseline_eval = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred_baseline,
        sensitive_features=sensitive_test,
        model_name="baseline",
    )

    fairness_for_bias = dict(baseline_eval["fairness"])
    fairness_for_bias["sensitive_group_distribution"] = (
        df[sensitive_column].value_counts(
            normalize=True, dropna=False).to_dict()
    )

    positive_label = _infer_positive_label(df[target_column])

    dataset_analysis = analyze_group_distribution_and_selection_rate(
        df=df,
        sensitive_attribute=sensitive_column,
        target_column=target_column,
        positive_label=positive_label,
    )
    correlation_analysis = analyze_feature_correlation_with_sensitive_attribute(
        df=df.drop(columns=[target_column], errors="ignore"),
        sensitive_attribute=sensitive_column,
        top_k=5,
    )

    bias_result = identify_bias_type(fairness_for_bias)
    diagnosis_result = diagnose_bias_root_causes(
        fairness_metrics=baseline_eval["fairness"],
        dataset_distribution_analysis=dataset_analysis,
        feature_correlation_analysis=correlation_analysis,
    )
    strategy_result = recommend_mitigation_strategies(
        bias_result["detected_bias_type"])
    recommended = strategy_result.get("recommended_strategies", [])
    recommended_strategies = [item["name"] for item in recommended] or [
        "Continuous Fairness Monitoring"
    ]

    simulation_results = simulate_mitigation_strategies(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        strategies=recommended_strategies,
        model=baseline_model,
        model_type=model_type,
        sensitive_column=sensitive_column,
        random_state=42,
    )
    comparison_report = compare_mitigation_strategies(simulation_results)

    _print_strategy_simulation_report(comparison_report)
    selected_strategy = _select_strategy_from_comparison(
        comparison_report=comparison_report,
        preferred_strategy=preferred_strategy,
        interactive=interactive_selection,
    )

    baseline_state = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "sensitive_train": sensitive_train,
        "sensitive_test": sensitive_test,
        "sensitive_column": sensitive_column,
        "baseline_model": baseline_model,
        "baseline_model_type": model_type,
        "y_pred_baseline": y_pred_baseline,
    }

    mitigated_model, y_pred_mitigated = apply_mitigation_strategy(
        strategy_name=selected_strategy,
        baseline_state=baseline_state,
    )

    comparison = compare_baseline_and_mitigated_models(
        y_true=y_test,
        y_pred_baseline=y_pred_baseline,
        y_pred_mitigated=y_pred_mitigated,
        sensitive_features=sensitive_test,
        baseline_name="baseline",
        mitigated_name=selected_strategy,
    )

    result_bundle = {
        "config": {
            "data_path": data_path,
            "target_column": target_column,
            "sensitive_column": sensitive_column,
            "model_type": model_type,
            "test_size": test_size,
            "selected_strategy": selected_strategy,
        },
        "bias_result": bias_result,
        "dataset_analysis": dataset_analysis,
        "correlation_analysis": correlation_analysis,
        "bias_diagnosis": diagnosis_result,
        "strategy_result": strategy_result,
        "strategy_simulation": simulation_results,
        "strategy_comparison": comparison_report,
        "selected_strategy": selected_strategy,
        "comparison": comparison,
    }

    model_artifacts = save_model_artifacts(
        baseline_model=baseline_model,
        mitigated_model=mitigated_model,
        selected_strategy=selected_strategy,
        target_column=target_column,
        sensitive_column=sensitive_column,
        model_type=model_type,
        output_paths=output_paths,
    )
    result_bundle["model_artifacts"] = model_artifacts

    # Save report artifacts for downstream dashboards or notebooks.
    reports_dir = output_paths["reports"]
    with (reports_dir / "pipeline_results.json").open("w", encoding="utf-8") as f:
        json.dump(result_bundle, f, indent=2)

    pd.DataFrame([comparison["baseline"]["performance"]]).to_csv(
        reports_dir / "baseline_performance.csv", index=False
    )
    pd.DataFrame([comparison["mitigated"]["performance"]]).to_csv(
        reports_dir / "mitigated_performance.csv", index=False
    )
    pd.DataFrame([comparison["baseline"]["fairness"]]).to_csv(
        reports_dir / "baseline_fairness.csv", index=False
    )
    pd.DataFrame([comparison["mitigated"]["fairness"]]).to_csv(
        reports_dir / "mitigated_fairness.csv", index=False
    )

    with (reports_dir / "bias_diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(diagnosis_result, f, indent=2)

    return result_bundle


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run baseline and fairness-mitigated hiring model pipeline."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to input CSV dataset. If omitted, pipeline tries data/*.csv.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. If omitted, pipeline tries common names.",
    )
    parser.add_argument(
        "--sensitive",
        default=None,
        help="Sensitive attribute column. If omitted, pipeline tries common names.",
    )
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
        help="Baseline model type.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (0-1).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory for reports and plots.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional mitigation strategy to apply after simulation. If omitted, the balanced choice is used or you can select interactively.",
    )
    return parser.parse_args()


def _resolve_data_path(data_arg: str | None) -> str:
    """Resolve data path from CLI arg or auto-detect from data directory."""
    if data_arg:
        path = Path(data_arg)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data_dir = Path("data")
    csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    if len(csv_files) == 1:
        return str(csv_files[0])
    if len(csv_files) > 1:
        file_names = ", ".join(str(p) for p in csv_files)
        raise ValueError(
            "Multiple CSV files found in data directory. Pass --data explicitly. "
            f"Found: {file_names}"
        )

    raise ValueError(
        "No dataset provided. Pass --data or place exactly one CSV in data/"
    )


def _infer_column(columns: list[str], preferred_names: list[str]) -> str | None:
    """Infer a column by checking common names case-insensitively."""
    normalized = {col.lower(): col for col in columns}
    for name in preferred_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _infer_positive_label(target_series: pd.Series) -> Any:
    """Infer a positive target label for reporting-oriented diagnostics."""
    values = target_series.dropna().unique().tolist()
    if len(values) != 2:
        return 1

    if set(values).issubset({0, 1}) or set(values).issubset({-1, 1}):
        return 1

    normalized = {str(value).strip().lower(): value for value in values}
    for candidate in ["1", "true", "yes", "y", "hired", "selected", "positive"]:
        if candidate in normalized:
            return normalized[candidate]

    return sorted(values, key=lambda value: str(value).lower())[-1]


def main() -> None:
    args = parse_args()

    try:
        data_path = _resolve_data_path(args.data)
        header_df = pd.read_csv(data_path, nrows=0)
        columns = list(header_df.columns)

        target_column = args.target or _infer_column(
            columns,
            ["target", "label", "hired", "hiring_outcome", "outcome"],
        )
        sensitive_column = args.sensitive or _infer_column(
            columns,
            ["gender", "race", "age", "sex", "ethnicity"],
        )

        if not target_column:
            raise ValueError(
                "Could not infer target column. Pass --target explicitly. "
                f"Available columns: {', '.join(columns)}"
            )
        if not sensitive_column:
            raise ValueError(
                "Could not infer sensitive column. Pass --sensitive explicitly. "
                f"Available columns: {', '.join(columns)}"
            )
    except Exception as exc:
        print(f"Input resolution error: {exc}")
        print(
            "Example: python main.py --data data/hiring.csv --target hired "
            "--sensitive gender"
        )
        raise SystemExit(2) from exc

    results = run_pipeline(
        data_path=data_path,
        target_column=target_column,
        sensitive_column=sensitive_column,
        model_type=args.model_type,
        test_size=args.test_size,
        output_dir=args.output_dir,
        preferred_strategy=args.strategy,
        interactive_selection=sys.stdin.isatty(),
    )

    print("Pipeline completed successfully.")
    print(f"Detected bias: {results['bias_result']['detected_bias_type']}")
    print("Bias diagnosis report:")
    print(json.dumps(results["bias_diagnosis"], indent=2))
    print(
        "Selected mitigation strategy: "
        f"{results['config']['selected_strategy']}"
    )
    print(f"Results saved under: {Path(args.output_dir).resolve()}")
    print("Saved model artifacts:")
    print(f"- Baseline: {results['model_artifacts']['baseline_model']}")
    print(f"- Mitigated: {results['model_artifacts']['mitigated_model']}")
    print(f"- Metadata: {results['model_artifacts']['metadata']}")


if __name__ == "__main__":
    main()
