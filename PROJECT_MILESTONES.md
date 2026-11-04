# IRAP Framework - Project Milestones & Documentation

This document tracks the progress, implementation details, and architectural decisions made throughout the M.Tech research project lifecycle. It serves as a living record to aid in thesis writing and defense preparation.

---

## 🟢 Week 1: Project Setup & Dataset Understanding Module
**Dates:** Aug 1 – Aug 7  
**Status:** Completed  

### Overview
Week 1 focused on establishing the foundational architecture of the Intelligent Relationship-Aware Preprocessing (IRAP) Framework. The goal was to build robust core models, implement data loading, and create the Dataset Understanding module that extracts high-level metadata before any column-level profiling occurs.

### Key Implementations

#### 1. Core Data Models (`src/models/`)
- **`Dataset` Object**: The central data hub that passes through the entire pipeline. It stores raw data, processed data, profiling results, relationship graphs, preprocessing plans, and pipeline history.
- **`DatasetProfile` Object**: Stores high-level dataset statistics (dimensions, memory usage, missing percentages, data type counts, target class distributions).
- **`FeatureProfile` Object**: A highly detailed container for column-level statistics, missing values, outliers, correlation scores, and preprocessing recommendations (to be populated in Week 2).

#### 2. Data Loading (`src/loader/dataset_loader.py`)
- Implemented robust file loading supporting `.csv`, `.xlsx`, and `.xls`.
- Integrated automated encoding detection (using `chardet`) with fallback mechanisms to prevent character corruption.
- Integrated automated separator detection (using `csv.Sniffer`) with fallback mechanisms for single-column datasets.

#### 3. Dataset Profiling (`src/profiling/dataset_profiler.py`)
- Implemented the `DatasetProfiler` to analyze the loaded dataset.
- Automatically calculates row/column counts, memory usage, duplicate percentages, and data type groupings.
- Generates crucial **Health Indicators**: `is_high_dimensional`, `has_many_missing_values`, `has_duplicates`, `has_constant_features`.
- Identifies target class imbalances if a `target_column` is defined.

#### 4. Orchestration & Logging (`src/main.py`)
- Built the primary execution script to tie the loader and profiler together.
- Established rigorous academic logging standard using Python's `logging` module. All decisions and pipeline steps are permanently recorded to `outputs/pipeline.log` to ensure absolute reproducibility.

### Architectural Research Enhancements
To ensure the framework meets M.Tech thesis standards and avoids common pitfalls, the following mechanisms were integrated:
- **Memory Scalability (OOM Prevention)**: Added a `get_sampled_data(max_rows)` method to the `Dataset` object using stratified sampling. This guarantees that computationally expensive mathematical operations (like SHAP interactions in Week 3/4) won't crash when running on larger datasets, explicitly addressing scalability concerns.

---

## 🟢 Week 2: Feature Profiling Module
**Dates:** Aug 8 – Aug 15  
**Status:** Completed  

### Overview
Week 2 focused on extracting highly granular, column-level statistics. This is critical for the Intelligent Decision Engine (Week 5), which relies on these deep mathematical properties (rather than just simple data types) to recommend robust preprocessing transformations.

### Key Implementations

#### 1. Feature Profiling (`src/profiling/feature_profiler.py`)
- **Numerical Analysis:** Automatically calculates robust statistics including Variance, Skewness, Kurtosis, and Interquartile Range (IQR). Detects basic outliers and explicitly tests for Gaussian distributions using D'Agostino's K-squared normality test.
- **Categorical Analysis:** Automatically calculates Shannon Entropy (Information Gain) and evaluates cardinality to prevent massive one-hot encoding matrices.
- **Scalability Preservation:** Wraps computations around the `Dataset.get_sampled_data()` method to guarantee OOM prevention during production loads.

#### 2. Pipeline Integration (`src/main.py`)
- Integrated `FeatureProfiler` into the main orchestration sequence.
- Validated execution using the synthetic testing framework.

---

## 🟢 Week 3 & 4: Relationship Discovery & Knowledge Graph Module
**Dates:** Aug 16 – Aug 31  
**Status:** Completed  

### Overview
Weeks 3 and 4 focused on implementing the core academic novelty of the IRAP framework: using explainable AI (SHAP) and surrogate modeling (XGBoost) to mathematically discover hidden, non-linear dependencies between dataset features and represent them as a Context Knowledge Graph (NetworkX).

### Key Implementations

#### 1. Relationship Discovery (`src/relationship/relationship_discovery.py`)
- **Surrogate Modeling:** Implemented a robust loop that isolates every feature as a target and trains an `XGBRegressor` or `XGBClassifier` to predict it using the remaining features.
- **Native NaN Handling:** Intelligently encoded categorical features using `pd.factorize` while carefully preserving `np.nan` values, allowing XGBoost to handle missing values natively during the relationship discovery phase.
- **SHAP Interaction Extraction:** Integrated `shap.TreeExplainer` to calculate the mean absolute SHAP values for every predictor. Automatically handles dimensionality differences between regression and multi-class classification outputs.
- **Scalability Preservation:** Computations run strictly on a sampled subset (`dataset.get_sampled_data()`) to guarantee execution without Out-Of-Memory (OOM) crashes, regardless of dataset size.

#### 2. Context Knowledge Graph
- Constructed a `networkx.DiGraph()` to store the relationships.
- Normalized SHAP scores to a 0–1 scale and applied a configurable threshold (`0.1`) to filter weak dependencies.
- Added weighted directed edges (`Predictor -> Target`) to the graph, forming the architectural foundation required for context-aware processing in later weeks.

#### 3. Pipeline Integration (`src/main.py`)
- Added Phase 4 to the main execution pipeline.
- Verified successful graph construction on the synthetic sample dataset.

---

## 🟢 Week 5: Intelligent Decision Engine
**Dates:** Sept 1 – Sept 7  
**Status:** Completed  

### Overview
Week 5 focused on building the "brain" of the IRAP framework. The Intelligent Decision Engine synthesizes the statistical profiles from Week 2 and the SHAP Knowledge Graph from Weeks 3/4 to generate a highly specific, deterministic preprocessing plan for every feature in the dataset.

### Key Implementations

#### 1. Rule-Based Decision Logic (`src/decision/decision_engine.py`)
- **Missing Value Strategy:** Automatically recommends dropping features with >50% missing data. For salvageable features, it queries the `relationship_graph`. If the feature has dependencies, it prescribes `ContextAwareKNN`; otherwise, it falls back to simple mathematical imputation (Median/Mode).
- **Categorical Encoding Strategy:** Evaluates the cardinality of string/object columns. If a category exceeds the threshold of 15 unique values (e.g., ICD-10 medical codes), it safely prevents memory explosion by routing to `TargetEncoding`/`FrequencyEncoding`. Smaller categorical columns are routed to `OneHotEncoding`.
- **Scaling and Outlier Strategy:** Evaluates skewness and outlier counts to select the safest scaling method (`RobustScaler` for highly skewed/outlier-heavy data, `StandardScaler` for Gaussian, and `MinMaxScaler` otherwise).

#### 2. Pipeline Integration (`src/main.py`)
- Added Phase 5 to the main execution pipeline.
- verified that a structured preprocessing plan is generated and successfully attached to `dataset.preprocessing_plan`.

---

## 🟢 Week 6: Context-Aware Missing Value Processing
**Dates:** Sept 8 – Sept 15  
**Status:** Completed  

### Overview
Week 6 implemented the Context-Aware Missing Value Imputation module (`src/preprocessing/imputer.py`), combining the discovered Knowledge Graph dependencies with the LOINC Biological Guardrails.

### Key Implementations
- **`ContextAwareKNN` Engine**: Selects nearest neighbors based strictly on the specific predictor graph edges discovered during Phase 4 SHAP surrogate modeling, preserving natural multivariate data topology.
- **LOINC Biological Guardrails Integration**: Queries `datasets/biological_guardrails.json` (40,778 clinical terms) to automatically clamp or reject physiologically impossible imputed values (e.g., negative blood pressures, negative cell counts, percentages outside $[0, 100]$).
- **Fallback Mechanisms**: Automatically falls back to median/mode imputation for independent features with no incoming graph edges.

---

## 🟢 Week 7: Outlier, Encoding, Scaling Modules
**Dates:** Sept 16 – Sept 23  
**Status:** Completed  

### Overview
Week 7 implemented adaptive transformation engines (`src/preprocessing/encoders.py`, `src/preprocessing/scalers.py`, `src/preprocessing/preprocessor.py`) to prevent dimensional explosion and outlier distortion.

### Key Implementations
- **Adaptive Categorical Encoding**:
  - Dynamically routes high-cardinality features (>15 categories, e.g. ICD-10 medical codes) to smoothed `TargetEncoding` or `FrequencyEncoding`, eliminating the curse of dimensionality.
  - Safely routes low-cardinality features to `OneHotEncoding`.
- **Adaptive Numerical Scaling**:
  - Applies `RobustScaler` (median & IQR) to skewed/outlier-heavy features to protect against extreme sensor spikes.
  - Applies `StandardScaler` to Gaussian/normally distributed features.
  - Applies `MinMaxScaler` to bounded features without heavy outliers.
- **`Preprocessor` Orchestrator**: Executes the complete plan and attaches the clean machine-learning-ready matrix to `dataset.processed_dataframe`.

---

## 🟢 Week 8: Validation Module & Ablation Studies
**Dates:** Sept 24 – Sept 30  
**Status:** Completed  

### Overview
Week 8 built the comparative validation engine (`src/validation/validator.py`) to empirically prove IRAP's superiority over standard preprocessing pipelines.

### Key Implementations
- **Controlled Ablation Benchmark**: Runs parallel downstream models (Random Forest) on both the Scikit-Learn baseline (`SimpleImputer` + `StandardScaler` + `OneHotEncoder`) and the IRAP pipeline (`ContextAwareKNN` + `AdaptiveEncoder` + `AdaptiveScaler`).
- **Comprehensive Metrics**: Computes Accuracy, F1-Score (weighted), Precision, Recall, and Imputation RMSE.
- **Ground-Truth Imputation Fidelity**: Artificially masks known physiological features and measures reconstruction RMSE, demonstrating up to a **44.06% reduction in imputation error** over standard mean imputation.

---

## 🟢 Week 9: Explainability Module
**Dates:** Oct 1 – Oct 10  
**Status:** Completed  

### Overview
Week 9 built the Explainable AI (XAI) engine (`src/explainability/explainer.py`), turning IRAP from a black-box into a fully auditable preprocessing framework.

### Key Implementations
- **Deterministic Rationale Cards**: Documents the exact mathematical reason for every imputation, encoding, and scaling decision per feature.
- **Graph Exports**: Automatically exports the Context Knowledge Graph to JSON (`outputs/knowledge_graph.json`) and DOT format (`outputs/knowledge_graph.dot`) for Graphviz and slide visualizers.

---

## 🟢 Week 10: Multi-Dataset Experiments & Benchmarking
**Dates:** Oct 11 – Oct 20  
**Status:** Completed  

### Overview
Week 10 executed automated comparative experiments across three diverse medical datasets (`experiments/run_benchmarks.py`).

### Key Implementations & Empirical Results
- **Realistic Synthetic Cohort (N=200)**: 44.06% imputation RMSE error reduction over mean imputation; successfully handled 48 ICD disease codes without memory explosion.
- **Mendeley CBC Hematology Dataset (N=374)**: IRAP achieved **97.80% Accuracy** vs **67.03% Baseline Accuracy** (**+30.77% gain**), and **97.81% F1-Score** vs **65.53% Baseline F1-Score** (**+32.28% gain**).
- **Hospital Diabetes Cohort (N=5,132)**: Scaled to 5,132 patients with 17.38% imputation RMSE reduction and 99.92% downstream predictive accuracy.
- Compiled summary tables to `outputs/reports/benchmark_summary.csv` and `outputs/reports/benchmark_summary.md`.

---

## 🟢 Week 11 & Buffer: Documentation, Reports & Defense Readiness
**Dates:** Oct 21 – Nov 7  
**Status:** Completed  

### Overview
Completed automated technical report generation (`src/reports/report_generator.py`), generating both Markdown and styled HTML executive reports (`outputs/reports/irap_execution_report.html`). Finalized slide materials and viva defense preparations in `PPT_PREPARATION.md`, `PANEL_REVIEW_QUESTIONS.md`, and `PROJECT_QNA.md`.
