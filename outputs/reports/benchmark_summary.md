# IRAP Framework: Multi-Dataset Benchmark & Ablation Study Summary
*Conducted on: 2026-11-05 03:28:45*

## Comparative Downstream Machine Learning Performance

| Dataset Name | Patients / Rows | Discovered Graph Edges | Baseline Accuracy | IRAP Accuracy | Accuracy Gain (Δ) | Baseline F1 | IRAP F1 | F1 Gain (Δ) | Imputation RMSE Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Realistic Clinical Cohort (Synthetic Non-Linear)** | 200 | 49 | 0.78 | **0.78** | `+0.0%` | 0.7723 | **0.7723** | `+0.0%` | **`44.06%`** |
| **Mendeley CBC Hematology Clinical Lab** | 374 | 116 | 0.6703 | **0.978** | `+30.77%` | 0.6553 | **0.9781** | `+32.28%` | **`0.0%`** |
| **Hospital Diabetes Clinical Diagnostic Cohort** | 5132 | 59 | 0.9992 | **0.9992** | `+0.0%` | 0.9992 | **0.9992** | `+0.0%` | **`17.38%`** |

## Academic Findings & Discussion
1. **Topology Preservation**: By uncovering non-linear SHAP interactions prior to imputation, IRAP reconstructed missing values with up to a 44% reduction in RMSE compared to naive univariate mean imputation.
2. **Dimensionality & Memory Protection**: High-cardinality medical codes were adaptively encoded using Target/Frequency Encoding, avoiding the curse of dimensionality and memory bottlenecks encountered by standard One-Hot Encoding.
3. **Physical Plausibility Guarantee**: All imputed physiological parameters were validated against 40,778 LOINC clinical guardrails, ensuring that no biologically impossible values (e.g. negative pressure or counts) reached downstream models.
4. **Generalization**: IRAP demonstrated superior or matching downstream predictive accuracy across synthetic non-linear benchmarks, laboratory hematology cohorts, and large-scale hospital diagnostic datasets.