"""Main pipeline for hiring fairness analysis and mitigation.

Example:
python main.py --data data/hiring.csv --target hired --sensitive gender
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump, load

from src.bias_detection.bias_identifier import identify_bias_type
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_dataset
from src.data.schema_validator import validate_dataset_schema
from src.mitigation.mitigation_methods import (
    apply_data_reweighing,
    train_fairness_constrained_model,
    train_threshold_optimizer,
)
from src.mitigation.strategy_recommender import recommend_mitigation_strategies
from src.models.evaluate_model import compare_baseline_and_mitigated_models, evaluate_predictions
from src.models.train_model import build_model, train_baseline_model, train_model


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


def run_pipeline(
        data_path: str,
        target_column: str,
        sensitive_column: str,
        model_type: str,
        test_size: float,
        output_dir: str,
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

    bias_result = identify_bias_type(fairness_for_bias)
    strategy_result = recommend_mitigation_strategies(
        bias_result["detected_bias_type"])
    recommended = strategy_result.get("recommended_strategies", [])
    selected_strategy = (
        recommended[0]["name"] if recommended else "Continuous Fairness Monitoring"
    )

    baseline_state = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "sensitive_train": sensitive_train,
        "sensitive_test": sensitive_test,
        "sensitive_column": sensitive_column,
        "baseline_model": baseline_model,
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
        "strategy_result": strategy_result,
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
    )

    print("Pipeline completed successfully.")
    print(f"Detected bias: {results['bias_result']['detected_bias_type']}")
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
