# IRAP Framework: Automated Technical & Clinical Preprocessing Report
*Generated on: 2026-11-05 03:28:30*

---

## 1. Executive Summary
- **Dataset File:** `Diabetes Classification.csv`
- **Raw Shape:** 5132 rows × 12 columns
- **Processed Shape:** (5132, 12)
- **Discovered Dependencies (Knowledge Graph Edges):** 59
- **Target Variable:** `Diagnosis` (Classification)

---

## 2. Dataset Health & Profiling
- **Global Missing Percentage:** 0.00% (0 cells)
- **Duplicate Rows:** 0 (0.00%)
- **Feature Breakdown:** 10 Numeric, 1 Categorical, 0 Boolean
- **Health Indicators Flagged:**
  - High Dimensionality: `False`
  - Heavy Missingness: `False`
  - Severe Outliers Present: `False`
  - High Cardinality Present: `False`

---

## 3. Discovered Knowledge Graph & Dependencies
Top non-linear feature relationships discovered via TreeSHAP interaction extraction:

| Predictor Feature | Target Feature | SHAP Interaction Weight | Clinical / Mathematical Meaning |
| :--- | :--- | :---: | :--- |
| **Unnamed: 0** | **Age** | `1.0000` | Direct functional dependence |
| **Unnamed: 0** | **BMI** | `1.0000` | Direct functional dependence |
| **Age** | **Unnamed: 0** | `1.0000` | Direct functional dependence |
| **Gender** | **Cr** | `1.0000` | Direct functional dependence |
| **Chol** | **TG** | `1.0000` | Direct functional dependence |
| **Chol** | **LDL** | `1.0000` | Direct functional dependence |
| **LDL** | **Chol** | `1.0000` | Direct functional dependence |
| **LDL** | **HDL** | `1.0000` | Direct functional dependence |
| **Cr** | **Gender** | `1.0000` | Direct functional dependence |
| **Cr** | **BUN** | `1.0000` | Direct functional dependence |

---

## 4. Intelligent Preprocessing Plan & Applied Actions
Deterministic preprocessing actions decided by the IRAP Engine:

| Feature | Missing % | Applied Imputation | Applied Encoding | Applied Scaling | Rationale |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Unnamed: 0** | 0.0% | None | None | `MinMaxScaler` | Feature is numeric without heavy outliers or strict normality. Defaulting to MinMaxScaler. |
| **Age** | 0.0% | None | None | `MinMaxScaler` | Feature is numeric without heavy outliers or strict normality. Defaulting to MinMaxScaler. |
| **Gender** | 0.0% | None | `OneHotEncoding` | None | Low cardinality (3 unique values). Safe for OneHotEncoding. |
| **BMI** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (151) or high skewness. RobustScaler prevents distortion. |
| **Chol** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (93) or high skewness. RobustScaler prevents distortion. |
| **TG** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (277) or high skewness. RobustScaler prevents distortion. |
| **HDL** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (463) or high skewness. RobustScaler prevents distortion. |
| **LDL** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (38) or high skewness. RobustScaler prevents distortion. |
| **Cr** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (97) or high skewness. RobustScaler prevents distortion. |
| **BUN** | 0.0% | None | None | `RobustScaler` | Feature is numeric with outliers (118) or high skewness. RobustScaler prevents distortion. |
| **Diagnosis** | 0.0% | None | None | `MinMaxScaler` | Feature is numeric without heavy outliers or strict normality. Defaulting to MinMaxScaler. |

---

## 5. Validation & Comparative Benchmarking (Ablation Study)
Downstream model performance comparing standard Scikit-Learn baseline vs IRAP:

| Evaluation Metric | Baseline (Scikit-Learn Standard) | IRAP Framework | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.9992 | **0.9992** | `+0.0%` |
| **F1-Score (Weighted)** | 0.9992 | **0.9992** | `+0.0%` |
| **Precision** | 0.9992 | **0.9992** | - |
| **Recall** | 0.9992 | **0.9992** | - |

### Imputation Fidelity (Ground Truth Reconstruction):
- **Tested Feature:** `Unnamed: 0`
- **Baseline Imputation RMSE:** `1517.1243`
- **IRAP Context-Aware Imputation RMSE:** `1253.5151`
- **Error Reduction:** **`17.38%` reduction in imputation error**

---

## 6. Execution Audit Trail
Full chronological log of pipeline execution:

- Dataset loaded successfully.
- Dataset profiled successfully.
- Feature profiling completed for 11 features.
- Relationship Discovery (SHAP + XGBoost) completed. Found 59 edges.
- Intelligent Decision Engine generated preprocessing plan.
- Context-Aware Imputation completed: 0 missing values filled across features. 0 biological guardrail bounds enforced.
- Adaptive Categorical Encoding completed: 1 feature(s) transformed.
- Adaptive Numerical Scaling completed: 9 feature(s) scaled.
- Preprocessing Phase completed. Final processed shape: (5132, 12).
- Pipeline Validation completed: Evaluated against Scikit-Learn baseline. Results stored in dataset.validation_results.
- Explainability Module completed: Generated deterministic audit cards for 11 features. Exported Context Knowledge Graph to outputs\knowledge_graph.dot.