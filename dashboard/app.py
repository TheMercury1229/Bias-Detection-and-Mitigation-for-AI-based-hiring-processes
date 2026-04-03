"""Streamlit dashboard for AI hiring fairness analysis and mitigation."""

from __future__ import annotations
from src.data.preprocess import preprocess_dataset
from src.data.schema_validator import validate_dataset_schema
from src.mitigation.mitigation_methods import (
    apply_data_reweighing,
    train_fairness_constrained_model,
    train_threshold_optimizer,
)
from src.mitigation.strategy_simulator import simulate_mitigation_strategies
from src.mitigation.strategy_recommender import recommend_mitigation_strategies
from src.mitigation.strategy_comparator import compare_mitigation_strategies
from src.models.train_model import build_model, train_baseline_model, train_model
from src.models.evaluate_model import (
    compare_baseline_and_mitigated_models,
    evaluate_predictions,
)
from src.explainability.shap_explainer import compute_shap_values
from src.explainability.group_explainer import analyze_group_shap_values
from src.explainability.bias_explainer import generate_bias_explanations
from src.bias_detection.bias_identifier import identify_bias_type

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _to_comparison_df(comparison: dict[str, Any], key: str) -> pd.DataFrame:
    baseline_data = comparison["baseline"][key]
    mitigated_data = comparison["mitigated"][key]

    rows = []
    for metric_name, baseline_value in baseline_data.items():
        mitigated_value = mitigated_data.get(metric_name)
        if isinstance(baseline_value, (dict, list)):
            continue
        rows.append(
            {
                "metric": metric_name,
                "baseline": float(baseline_value),
                "mitigated": float(mitigated_value),
                "delta": float(mitigated_value) - float(baseline_value),
            }
        )

    return pd.DataFrame(rows)


def _display_comparison_charts(comparison: dict[str, Any]) -> None:
    st.subheader("Model Comparison")

    perf_df = _to_comparison_df(comparison, "performance")
    fair_df = _to_comparison_df(comparison, "fairness")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Performance Metrics")
        st.dataframe(perf_df, use_container_width=True)
        if not perf_df.empty:
            chart_df = perf_df.set_index("metric")[["baseline", "mitigated"]]
            st.bar_chart(chart_df)

    with col2:
        st.markdown("### Fairness Metrics")
        st.dataframe(fair_df, use_container_width=True)
        if not fair_df.empty:
            chart_df = fair_df.set_index("metric")[["baseline", "mitigated"]]
            st.bar_chart(chart_df)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("### Baseline Confusion Matrix")
        st.dataframe(
            pd.DataFrame(comparison["baseline"]
                         ["performance"]["confusion_matrix"]),
            use_container_width=True,
        )
    with cm_col2:
        st.markdown("### Mitigated Confusion Matrix")
        st.dataframe(
            pd.DataFrame(comparison["mitigated"]
                         ["performance"]["confusion_matrix"]),
            use_container_width=True,
        )

    st.markdown("### Fairness Improvement Summary")
    delta_df = pd.DataFrame([comparison["delta"]]).T.reset_index()
    delta_df.columns = ["metric", "change"]
    st.dataframe(delta_df, use_container_width=True)


def _run_strategy_simulation(baseline_state: dict[str, Any]) -> dict[str, Any]:
    """Run and rank mitigation strategy simulations for the current dataset."""
    recommended = baseline_state["strategy_result"].get(
        "recommended_strategies", [])
    strategy_names = [item["name"] for item in recommended] or [
        "Continuous Fairness Monitoring"
    ]

    simulation_results = simulate_mitigation_strategies(
        X_train=baseline_state["X_train"],
        X_test=baseline_state["X_test"],
        y_train=baseline_state["y_train"],
        y_test=baseline_state["y_test"],
        sensitive_train=baseline_state["sensitive_train"],
        sensitive_test=baseline_state["sensitive_test"],
        strategies=strategy_names,
        model=baseline_state["baseline_model"],
        model_type=baseline_state["baseline_model_type"],
        sensitive_column=baseline_state["sensitive_column"],
        random_state=42,
    )
    return compare_mitigation_strategies(simulation_results)


def _display_explainability_section(
        model: Any,
        X: pd.DataFrame,
        sensitive_features: pd.Series,
        sensitive_column: str,
) -> None:
    """Render SHAP-based explainability views and bias insights."""
    st.subheader("Explainability")
    st.caption(
        "This section shows which features most influenced the model overall, "
        "and how those influences differ across demographic groups."
    )

    if not hasattr(model, "estimators_"):
        st.warning(
            "SHAP explanations are available only for tree-based models. "
            "Select Random Forest as the baseline model to view explainability results."
        )
        return

    try:
        shap_result = compute_shap_values(
            model=model, X=X, feature_names=list(X.columns))
        shap_values = pd.DataFrame(
            shap_result["shap_values"], columns=shap_result["feature_names"]
        )
        overall_importance = shap_values.abs().mean().sort_values(ascending=False)

        st.markdown("### Overall Feature Importance")
        importance_df = overall_importance.head(
            10).rename("mean_abs_shap").reset_index()
        importance_df.columns = ["feature", "mean_abs_shap"]
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.set_index("feature"))

        group_comparison = analyze_group_shap_values(
            model=model,
            X=X,
            sensitive_features=sensitive_features,
            feature_names=list(X.columns),
        )

        st.markdown("### Group-wise SHAP Comparison")
        comparison_df = pd.DataFrame(group_comparison).T
        if not comparison_df.empty:
            group_spread = (comparison_df.max(axis=0) - comparison_df.min(axis=0)).sort_values(
                ascending=False
            )
            top_features = group_spread.head(10).index.tolist()
            comparison_subset = comparison_df[top_features].T
            st.dataframe(comparison_df, use_container_width=True)
            st.bar_chart(comparison_subset)
        else:
            st.info("No group-level SHAP comparison could be generated.")

        bias_insights = generate_bias_explanations(group_comparison)
        st.markdown("### Human-Readable Bias Insights")
        for insight in bias_insights["insights"]:
            st.write(f"- {insight}")

        st.markdown(
            "The charts above help identify whether the model relies on certain "
            f"features more heavily for {sensitive_column} groups, which may indicate proxy bias or unequal treatment."
        )
    except Exception as exc:
        st.error(f"Explainability analysis failed: {exc}")


def _run_baseline_analysis(
        df: pd.DataFrame,
        target_column: str,
        sensitive_column: str,
    model_type: str,
        test_size: float,
) -> dict[str, Any]:
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

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": sensitive_train,
        "sensitive_test": sensitive_test,
        "baseline_model": baseline_model,
        "baseline_model_type": model_type,
        "y_pred_baseline": y_pred_baseline,
        "baseline_eval": baseline_eval,
        "bias_result": bias_result,
        "strategy_result": strategy_result,
        "sensitive_column": sensitive_column,
    }


def _apply_selected_mitigation(
        strategy_name: str,
        baseline_state: dict[str, Any],
) -> tuple[Any, Any]:
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
        base_estimator = build_model(
            model_type="logistic_regression",
            random_state=42,
        )
        constraints = (
            "equalized_odds"
            if strategy_name == "Equalized Odds Post-Processing"
            else "demographic_parity"
        )
        model = train_threshold_optimizer(
            estimator=base_estimator,
            X_train=X_train,
            y_train=y_train,
            sensitive_features=sensitive_train,
            constraints=constraints,
            predict_method="predict_proba",
        )
        y_pred = model.predict(X_test, sensitive_features=sensitive_test)
        return model, y_pred

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

    # Monitoring/no-op strategy fallback.
    model = baseline_state["baseline_model"]
    y_pred = baseline_state["y_pred_baseline"]
    return model, y_pred


def main() -> None:
    st.set_page_config(
        page_title="AI Hiring Fairness Dashboard", layout="wide")
    st.title("AI Hiring Fairness Dashboard")
    st.write(
        "Upload a hiring dataset, detect bias, and compare baseline vs fairness-"
        "mitigated model performance."
    )

    uploaded_file = st.file_uploader(
        "Upload hiring dataset (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to begin analysis.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read uploaded file: {exc}")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if df.empty:
        st.error("The uploaded dataset is empty.")
        return

    columns = list(df.columns)
    target_column = st.selectbox(
        "Select target column (hiring outcome)", columns)

    sensitive_candidates = [col for col in columns if col != target_column]
    if not sensitive_candidates:
        st.error(
            "Dataset needs at least one non-target column for sensitive attribute.")
        return

    sensitive_column = st.selectbox(
        "Select sensitive attribute column",
        sensitive_candidates,
    )
    model_type = st.selectbox(
        "Select baseline model type for explainability",
        options=["random_forest", "logistic_regression"],
        format_func=lambda value: "Random Forest" if value == "random_forest" else "Logistic Regression",
        index=0,
    )
    test_size = st.slider("Test size", min_value=0.1,
                          max_value=0.5, value=0.2, step=0.05)

    if st.button("Run Bias Detection Analysis", type="primary"):
        try:
            baseline_state = _run_baseline_analysis(
                df=df,
                target_column=target_column,
                sensitive_column=sensitive_column,
                model_type=model_type,
                test_size=test_size,
            )
            st.session_state["baseline_state"] = baseline_state
            st.session_state["strategy_comparison"] = _run_strategy_simulation(
                baseline_state=baseline_state,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    if "baseline_state" not in st.session_state:
        return

    baseline_state = st.session_state["baseline_state"]
    baseline_eval = baseline_state["baseline_eval"]
    bias_result = baseline_state["bias_result"]
    strategy_result = baseline_state["strategy_result"]
    strategy_comparison = st.session_state["strategy_comparison"]

    st.subheader("Fairness Metrics (Baseline)")
    st.dataframe(
        pd.DataFrame([baseline_eval["fairness"]]).drop(
            columns=["selection_rate_by_group"]),
        use_container_width=True,
    )
    st.write("Selection Rate by Group")
    st.table(pd.DataFrame(baseline_eval["fairness"]["selection_rate_by_group"].items(
    ), columns=["group", "selection_rate"]))

    st.subheader("Detected Bias Type")
    st.write(f"Bias Type: {bias_result['detected_bias_type']}")
    st.write(f"Severity Score: {bias_result['severity_score']}")
    st.write(bias_result["explanation"])

    _display_explainability_section(
        model=baseline_state["baseline_model"],
        X=baseline_state["X_test"],
        sensitive_features=baseline_state["sensitive_test"],
        sensitive_column=baseline_state["sensitive_column"],
    )

    st.subheader("Recommended Mitigation Strategies")
    strategy_table = pd.DataFrame(strategy_result["recommended_strategies"])
    st.table(strategy_table)

    st.subheader("Strategy Simulation")
    st.caption(
        "Compare fairness and accuracy before retraining. Lower fairness scores are better."
    )
    simulation_df = pd.DataFrame(strategy_comparison["sorted_results"])
    if not simulation_df.empty:
        display_columns = [
            column
            for column in [
                "strategy",
                "accuracy",
                "fairness",
                "fairness_improvement",
                "accuracy_retention",
                "rank_score",
            ]
            if column in simulation_df.columns
        ]
        st.dataframe(simulation_df[display_columns], use_container_width=True)
        chart_columns = [column for column in ["accuracy",
                                               "fairness"] if column in simulation_df.columns]
        if chart_columns:
            st.bar_chart(simulation_df.set_index("strategy")[chart_columns])

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Best fairness",
                strategy_comparison["best_fairness_strategy"]["strategy"],
            )
        with metric_col2:
            st.metric(
                "Best accuracy",
                strategy_comparison["best_accuracy_strategy"]["strategy"],
            )
        with metric_col3:
            st.metric(
                "Balanced choice",
                strategy_comparison["balanced_choice"]["strategy"],
            )
    else:
        st.info("No strategy simulation results available.")

    strategy_names = simulation_df["strategy"].tolist(
    ) if not simulation_df.empty else []
    if not strategy_names:
        st.warning(
            "No mitigation strategies available for the detected bias type.")
        return

    default_strategy = strategy_comparison["balanced_choice"]["strategy"]
    default_index = (
        strategy_names.index(default_strategy)
        if default_strategy in strategy_names
        else 0
    )
    selected_strategy = st.selectbox(
        "Select mitigation strategy",
        strategy_names,
        index=default_index,
    )

    if st.button("Train Fairness-Mitigated Model"):
        try:
            mitigated_model, y_pred_mitigated = _apply_selected_mitigation(
                strategy_name=selected_strategy,
                baseline_state=baseline_state,
            )

            comparison = compare_baseline_and_mitigated_models(
                y_true=baseline_state["y_test"],
                y_pred_baseline=baseline_state["y_pred_baseline"],
                y_pred_mitigated=y_pred_mitigated,
                sensitive_features=baseline_state["sensitive_test"],
                baseline_name="baseline",
                mitigated_name=selected_strategy,
            )

            st.session_state["mitigated_model"] = mitigated_model
            st.session_state["comparison"] = comparison
        except Exception as exc:
            st.error(f"Mitigated training failed: {exc}")
            return

    if "comparison" in st.session_state:
        _display_comparison_charts(st.session_state["comparison"])


if __name__ == "__main__":
    main()
