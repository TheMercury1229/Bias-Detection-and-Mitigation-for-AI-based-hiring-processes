# AI Fairness & Bias Mitigation Toolkit

An end-to-end framework for detecting, analyzing, and mitigating algorithmic bias in machine learning models, specifically tailored for HR and recruitment datasets (e.g., hiring, promotions).

## Project Overview

This project addresses the growing concern of "Black Box" AI discrimination. It provides tools to:

1.  **Identify Bias**: Detect if a model or dataset unfairly favors specific groups based on protected attributes (Gender, Ethnicity, Age).
2.  **Quantify Fairness**: Use industry-standard metrics like Disparate Impact, Equal Opportunity Difference, and Demographic Parity.
3.  **Mitigate Inequity**: Apply state-of-the-art algorithms (Pre-processing, In-processing, and Post-processing) to reduce bias while maintaining model performance.
4.  **Explain Decisions**: Leverage SHAP analysis to understand which features drive biased outcomes.

---

## Current Implementation

### 1. Bias Detection Engine (`src/bias_detection/`)

- **Metrics**: Implementation of `Disparate Impact Ratio`, `Demographic Parity Difference`, and `Equal Opportunity Difference`.
- **Automated Identifier**: A rule-based system that classifies the _type_ of bias found (e.g., "Historical Representation Bias" vs "Systemic Selection Bias") and suggests whether the bias is "Critical" or "Moderate."
- **Data Diagnostics**: Group-distribution and selection-rate analysis to detect historical imbalance and outcome disparities.
- **Correlation Analyzer**: Detects proxy bias risk by identifying features strongly correlated with sensitive attributes.
- **Root-Cause Diagnosis**: Combines fairness metrics + distribution + correlations to explain why bias is happening.

### 2. Mitigation Strategies (`src/mitigation/`)

- **Pre-processing**: `Reweighing` (AIF360) to adjust sample weights before training.
- **In-processing**: `Exponentiated Gradient` (Fairlearn) to optimize for fairness constraints during model training.
- **Post-processing**: `Threshold Optimizer` to adjust classification boundaries after training.
- **Strategy Recommender**: Logic-based assistant that recommends the best mitigation method based on the detected bias level and data availability.
- **Strategy Simulator**: Simulates recommended strategies before final retraining to estimate fairness/accuracy impact.
- **Strategy Comparator**: Ranks simulated strategies by fairness improvement and accuracy retention.

### 3. Model Training Pipeline (`src/models/`)

- **Framework**: Scikit-learn integration for `Logistic Regression` and `Random Forest`.
- **Fairness-Aware Training**: Centralized module that handles standard training and fairness-constrained optimization seamlessly.
- **Persistence**: Automatic saving and loading of model artifacts using `joblib`.

### 4. Interactive Dashboard (`dashboard/`)

- **Streamlit UI**: A user-friendly interface to:
  - Upload custom datasets.
  - Visualize bias distributions.
  - Compare "Baseline" vs "Mitigated" model performance and fairness.
  - Explore SHAP feature importance and group-wise explainability.
  - Review readable bias insights for non-technical stakeholders.
  - Compare mitigation strategy simulations and choose a final strategy from fairness-accuracy trade-offs.

### 5. CLI Pipeline (`main.py`)

- An automated end-to-end script that:
  - Infers target/protected columns automatically.
  - Detects bias in the raw data and generates root-cause diagnosis.
  - Recommends strategies, simulates them, and compares results.
  - Trains a baseline model.
  - Applies the selected mitigation strategy.
  - Saves results to `results/`.

---

## Scope & Implementation Level

- **Domain**: Optimized for tabular classification data (HR, Finance, Legal).
- **Integration Level**: **Advanced Prototype / V1.0**. The core library is modular, allowing developers to use individual components (`src/`) or the full pipeline.
- **Data Support**: Currently supports CSV files with explicit protected attributes.
- **Models Supported**: Built-in support for any Scikit-learn classifier through the `build_model` wrapper.
- **Explainability Scope**: SHAP explainability is fully supported for tree-based models (for example, Random Forest).

---

## Getting Started

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd inhouse
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

- **Run the CLI Pipeline**:
  ```bash
  python main.py --data data/hiring_data.csv
  ```
- **Launch the Dashboard**:
  ```bash
  streamlit run dashboard/app.py
  ```
- **Generate Test Data**:

  ```bash
  python data/generate-dataset.py
  ```

- **Generate Larger, More Obvious Bias Data**:
  ```bash
  python data/generate-dataset.py --rows 8000 --output data/hiring_synthetic_biased.csv
  ```

## Output Artifacts

- `results/reports/pipeline_results.json`: Full pipeline output bundle.
- `results/reports/bias_diagnosis.json`: Structured root-cause diagnosis.
- `results/models/baseline_model.joblib`: Baseline trained model.
- `results/models/mitigated_model.joblib`: Mitigated trained model.
- `results/models/model_metadata.json`: Model and strategy metadata.

## Project Concept

The core idea is to move beyond simple "fairness awareness" into **automated remediation**. Most tools only tell you _that_ you have a problem; this toolkit suggests _how_ to fix it and provides the code to execute the fix in a single workflow.
