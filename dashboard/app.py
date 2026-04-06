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
import pandas as pd
import streamlit as st
import os
import warnings
from typing import Any
from pathlib import Path
import sys

# Set runtime env and warning filters before importing project modules.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"inFairness.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"keras.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_binary_target(series: pd.Series) -> bool:
    values = pd.Series(series).dropna().unique().tolist()
    return len(values) == 2


def _safe_sensitive_candidates(df: pd.DataFrame, target_column: str) -> list[str]:
    """Prefer columns suitable for group fairness metrics (low/medium cardinality)."""
    candidates: list[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        unique_count = int(df[col].nunique(dropna=True))
        if unique_count <= 20:
            candidates.append(col)
    return candidates


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
        st.dataframe(perf_df, width='stretch')
        if not perf_df.empty:
            chart_df = perf_df.set_index("metric")[["baseline", "mitigated"]]
            st.bar_chart(chart_df)

    with col2:
        st.markdown("### Fairness Metrics")
        st.dataframe(fair_df, width='stretch')
        if not fair_df.empty:
            chart_df = fair_df.set_index("metric")[["baseline", "mitigated"]]
            st.bar_chart(chart_df)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("### Baseline Confusion Matrix")
        st.dataframe(
            pd.DataFrame(comparison["baseline"]
                         ["performance"]["confusion_matrix"]),
            width='stretch',
        )
    with cm_col2:
        st.markdown("### Mitigated Confusion Matrix")
        st.dataframe(
            pd.DataFrame(comparison["mitigated"]
                         ["performance"]["confusion_matrix"]),
            width='stretch',
        )

    st.markdown("### Fairness Improvement Summary")
    delta_df = pd.DataFrame([comparison["delta"]]).T.reset_index()
    delta_df.columns = ["metric", "change"]
    st.dataframe(delta_df, width='stretch')


def _display_benchmark_plots(benchmark_df: pd.DataFrame) -> None:
    """Show baseline model benchmark charts to justify model selection."""
    if benchmark_df.empty:
        return

    st.markdown("### Baseline Model Benchmark")
    st.caption(
        "Compares unmitigated model candidates before fairness mitigation. "
        "Lower fairness score is better."
    )
    st.dataframe(benchmark_df, width='stretch')

    chart_cols = [col for col in ["accuracy", "fairness_score"]
                  if col in benchmark_df.columns]
    if len(chart_cols) == 2:
        scatter_df = benchmark_df.set_index("model_name")
        st.scatter_chart(scatter_df[["accuracy", "fairness_score"]])

    value_cols = [col for col in ["accuracy", "demographic_parity_difference",
                                  "equalized_odds_difference"] if col in benchmark_df.columns]
    if value_cols:
        st.bar_chart(benchmark_df.set_index("model_name")[value_cols])


def _display_strategy_tradeoff_plots(simulation_df: pd.DataFrame) -> None:
    """Visualize mitigation trade-offs and ranking signals."""
    if simulation_df.empty:
        return

    st.markdown("### Mitigation Trade-off Plots")
    tradeoff_cols = [col for col in ["accuracy", "fairness"]
                     if col in simulation_df.columns]
    if len(tradeoff_cols) == 2:
        st.scatter_chart(simulation_df.set_index("strategy")[tradeoff_cols])

    signal_cols = [
        col
        for col in ["fairness_improvement", "accuracy_retention", "rank_score"]
        if col in simulation_df.columns
    ]
    if signal_cols:
        st.bar_chart(simulation_df.set_index("strategy")[signal_cols])


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
    model_type: str | None = None,
) -> None:
    """Render SHAP-based explainability views and bias insights."""
    st.subheader("Explainability")
    st.caption(
        "This section shows which features most influenced the model overall, "
        "and how those influences differ across demographic groups."
    )

    if not hasattr(model, "estimators_"):
        current_model = model_type or type(model).__name__
        st.warning(
            f"⚠️ SHAP feature importance is unavailable.\n\n"
            f"**Current model:** {current_model}\n\n"
            f"SHAP TreeExplainer requires tree-based models. "
            f"To enable this analysis, select **'Random Forest'** as the baseline model type. "
            f"Logistic Regression and linear models use different explainability methods (e.g., LIME)."
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
        st.dataframe(importance_df, width='stretch')
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
            st.dataframe(comparison_df, width='stretch')
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
    predicted_positive_rate = float(pd.Series(y_pred_baseline).mean())

    benchmark_rows: list[dict[str, Any]] = []
    for candidate_model in ["logistic_regression", "random_forest"]:
        try:
            candidate = train_baseline_model(
                X_train=X_train,
                y_train=y_train,
                model_type=candidate_model,
                random_state=42,
            )
            candidate_pred = candidate.predict(X_test)
            candidate_eval = evaluate_predictions(
                y_true=y_test,
                y_pred=candidate_pred,
                sensitive_features=sensitive_test,
                model_name=candidate_model,
            )
            fairness = candidate_eval["fairness"]
            fairness_score = (
                abs(float(fairness.get("demographic_parity_difference", 0.0)))
                + abs(float(fairness.get("equalized_odds_difference", 0.0)))
                + abs(1.0 - float(fairness.get("disparate_impact_ratio", 1.0)))
            ) / 3.0
            benchmark_rows.append(
                {
                    "model_name": candidate_model,
                    "accuracy": float(candidate_eval["performance"].get("accuracy", 0.0)),
                    "demographic_parity_difference": float(fairness.get("demographic_parity_difference", 0.0)),
                    "equalized_odds_difference": float(fairness.get("equalized_odds_difference", 0.0)),
                    "disparate_impact_ratio": float(fairness.get("disparate_impact_ratio", 1.0)),
                    "fairness_score": float(round(fairness_score, 6)),
                }
            )
        except Exception:
            # Keep dashboard responsive if one benchmark model fails.
            continue

    unique_labels = set(pd.Series(y_train).dropna().unique().tolist())
    is_binary_target_for_mitigation = unique_labels.issubset({0, 1})

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
        "is_binary_target_for_mitigation": is_binary_target_for_mitigation,
        "benchmark_df": pd.DataFrame(benchmark_rows),
        "predicted_positive_rate": predicted_positive_rate,
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

    if not baseline_state.get("is_binary_target_for_mitigation", True):
        model = baseline_state["baseline_model"]
        y_pred = baseline_state["y_pred_baseline"]
        return model, y_pred

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


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
            .module-card {
                border: 1px solid rgba(49, 83, 109, 0.22);
                border-radius: 14px;
                padding: 14px;
                background: linear-gradient(140deg, rgba(245,250,255,0.95), rgba(236,246,253,0.85));
                min-height: 118px;
            }
            .module-title {
                margin: 0;
                color: #0f2f4f;
                font-weight: 700;
                font-size: 1rem;
            }
            .module-text {
                margin-top: 6px;
                color: #2f455a;
                font-size: 0.9rem;
                line-height: 1.35rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                border: 1px solid rgba(49, 83, 109, 0.22);
                background: #f4f8fc;
                padding: 0.45rem 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_module_overview() -> None:
    st.caption(
        "Module-oriented workflow: Bias Detection -> Explainability -> Mitigation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="module-card">
                <p class="module-title">1) Bias Detection</p>
                <p class="module-text">Fairness metrics, selection-rate analysis, and bias-type identification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="module-card">
                <p class="module-title">2) Explainability</p>
                <p class="module-text">SHAP feature influence and group-wise contribution analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="module-card">
                <p class="module-title">3) Mitigation</p>
                <p class="module-text">Strategy simulation, trade-off ranking, and mitigated model comparison.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Hiring Fairness Dashboard", layout="wide")
    _inject_dashboard_styles()

    st.title("AI Hiring Fairness Dashboard")
    st.write(
        "Upload a hiring dataset, detect bias, and compare baseline vs fairness-"
        "mitigated model performance."
    )
    _render_module_overview()

    with st.sidebar:
        st.header("Pipeline Controls")
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

    if df.empty:
        st.error("The uploaded dataset is empty.")
        return

    columns = list(df.columns)
    with st.sidebar:
        target_column = st.selectbox("Target column (hiring outcome)", columns)

    sensitive_candidates = _safe_sensitive_candidates(df, target_column)
    if not sensitive_candidates:
        st.error(
            "No suitable sensitive attribute found with <= 20 unique values. "
            "Choose a categorical group column (for example gender, ethnicity, age_group)."
        )
        return

    with st.sidebar:
        sensitive_column = st.selectbox(
            "Sensitive attribute column",
            sensitive_candidates,
        )
        model_type = st.selectbox(
            "Baseline model type",
            options=["random_forest", "logistic_regression"],
            format_func=lambda value: "Random Forest" if value == "random_forest" else "Logistic Regression",
            index=0,
        )
        test_size = st.slider("Test size", min_value=0.1,
                              max_value=0.5, value=0.2, step=0.05)
        run_clicked = st.button("Run Bias Detection Analysis", type="primary")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), width='stretch')

    if not _is_binary_target(df[target_column]):
        st.error(
            "Selected target column is not binary. Fairness metrics and mitigation in this dashboard "
            "require a binary outcome (exactly 2 unique labels)."
        )
        st.info(
            "Tip: pick a binary target such as hired/selected, or pre-convert your target labels to two classes."
        )
        return

    sensitive_unique = int(df[sensitive_column].nunique(dropna=True))
    if sensitive_unique > 20:
        st.warning(
            f"Selected sensitive attribute has {sensitive_unique} unique groups. "
            "Fairness metrics may be noisy; use a grouped categorical column for clearer analysis."
        )

    if run_clicked:
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
            st.session_state.pop("comparison", None)
            st.session_state.pop("mitigated_model", None)
        except MemoryError as exc:
            st.error(
                "Analysis failed due to memory limits during strategy simulation. "
                "Try selecting 'Logistic Regression' as baseline model type or use a smaller test size."
            )
            st.caption(f"Details: {exc}")
            return
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
    benchmark_df = baseline_state.get("benchmark_df", pd.DataFrame())
    predicted_positive_rate = float(
        baseline_state.get("predicted_positive_rate", 0.0)
    )

    tab_bias, tab_explain, tab_mitigate = st.tabs(
        ["Bias Detection", "Explainability", "Mitigation & Comparison"]
    )

    with tab_bias:
        st.subheader("Fairness Metrics (Baseline)")
        if predicted_positive_rate < 0.02:
            st.warning(
                f"Very low predicted positive rate ({predicted_positive_rate:.2%}). "
                "Model may be collapsing to mostly negative predictions, making fairness comparison less reliable."
            )

        selection_rate_by_group = baseline_eval["fairness"].get(
            "selection_rate_by_group", {})
        if selection_rate_by_group and max(selection_rate_by_group.values()) == 0:
            st.warning(
                "All groups have zero selection rate. Fairness metrics are currently not informative "
                "for mitigation comparison in this run."
            )

        fairness_df = pd.DataFrame([baseline_eval["fairness"]])
        if "selection_rate_by_group" in fairness_df.columns:
            fairness_df = fairness_df.drop(columns=["selection_rate_by_group"])
        st.dataframe(fairness_df, width='stretch')

        st.write("Selection Rate by Group")
        selection_rate_items = selection_rate_by_group.items()
        st.table(pd.DataFrame(selection_rate_items,
                 columns=["group", "selection_rate"]))

        st.subheader("Detected Bias Type")
        st.write(f"Bias Type: {bias_result['detected_bias_type']}")
        st.write(f"Severity Score: {bias_result['severity_score']}")
        st.write(bias_result["explanation"])
        _display_benchmark_plots(benchmark_df)

    with tab_explain:
        _display_explainability_section(
            model=baseline_state["baseline_model"],
            X=baseline_state["X_test"],
            sensitive_features=baseline_state["sensitive_test"],
            sensitive_column=baseline_state["sensitive_column"],
            model_type=baseline_state.get("baseline_model_type"),
        )

    with tab_mitigate:
        if not baseline_state.get("is_binary_target_for_mitigation", True):
            st.warning(
                "Mitigation methods that rely on Fairlearn reductions/post-processing were skipped "
                "because the selected target is not binary 0/1. Baseline analysis and diagnostics are still shown."
            )

        st.subheader("Recommended Mitigation Strategies")
        strategy_table = pd.DataFrame(
            strategy_result["recommended_strategies"])
        if strategy_table.empty:
            st.info("No explicit mitigation strategy was recommended for this run.")
        else:
            st.table(strategy_table)

        st.subheader("Strategy Simulation")
        st.caption(
            "Compare fairness and accuracy before retraining. Lower fairness scores are better."
        )
        simulation_df = pd.DataFrame(
            strategy_comparison.get("sorted_results", []))
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
            st.dataframe(simulation_df[display_columns], width='stretch')
            chart_columns = [column for column in ["accuracy",
                                                   "fairness"] if column in simulation_df.columns]
            if chart_columns:
                st.bar_chart(simulation_df.set_index(
                    "strategy")[chart_columns])

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(
                    "Best fairness",
                    strategy_comparison["best_fairness_strategy"].get(
                        "strategy") or "N/A",
                )
            with metric_col2:
                st.metric(
                    "Best accuracy",
                    strategy_comparison["best_accuracy_strategy"].get(
                        "strategy") or "N/A",
                )
            with metric_col3:
                st.metric(
                    "Balanced choice",
                    strategy_comparison["balanced_choice"].get(
                        "strategy") or "N/A",
                )
            _display_strategy_tradeoff_plots(simulation_df)
        else:
            st.warning(
                "No strategy simulation results were produced. Falling back to continuous monitoring."
            )

        strategy_names = simulation_df["strategy"].tolist(
        ) if not simulation_df.empty else []
        if not strategy_names:
            strategy_names = ["Continuous Fairness Monitoring"]

        default_strategy = strategy_comparison.get(
            "balanced_choice", {}).get("strategy")
        if default_strategy not in strategy_names:
            default_strategy = strategy_names[0]
        default_index = strategy_names.index(default_strategy)

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
