# IRAP Framework 

**Intelligent Relationship-Aware Preprocessing Framework**

An advanced data preprocessing pipeline for M.Tech research that prioritizes understanding the dataset and discovering hidden mathematical relationships *before* making any preprocessing decisions. 

## Project Philosophy
Standard preprocessing often applies blind statistical methods (like mean imputation) without understanding the underlying data dependencies. The IRAP framework follows a structured cognitive approach:

`Observe` ➔ `Understand` ➔ `Learn Relationships` ➔ `Reason` ➔ `Preprocess` ➔ `Validate` ➔ `Explain`

## Key Capabilities
*   **Memory Scalable Design:** Utilizes intelligent stratified sampling to prevent Out-Of-Memory (OOM) errors during heavy O(N^2) mathematical operations on large datasets.
*   **Dataset Understanding:** Extracts deep metadata, statistics, and health indicators from raw datasets.
*   **Relationship Discovery (Upcoming):** Finds linear, non-linear, and hidden feature interactions using advanced statistical measures and explainable AI techniques (e.g., SHAP).
*   **Intelligent Decision Engine (Upcoming):** Recommends context-aware imputations and encodings based on discovered relationships rather than naive column statistics.
*   **Reproducible Research:** Generates highly detailed execution logs (`outputs/pipeline.log`) for every step to guarantee academic reproducibility.

## Project Structure
```
IRAP_FRAMEWORK/
├── datasets/            # Contains sample and synthetic datasets
├── outputs/             # Pipeline execution logs and generated artifacts
├── src/
│   ├── loader/          # Robust dataset ingestion (CSV, Excel)
│   ├── models/          # Core dataclasses (Dataset, DatasetProfile, FeatureProfile)
│   ├── profiling/       # Metadata and feature statistics extraction
│   └── main.py          # Framework orchestrator and entry point
├── requirements.txt     # Python dependencies
├── PROJECT_MILESTONES.md # Detailed progress tracker and weekly deliverables
└── README.md
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline orchestrator:
   ```bash
   python src/main.py
   ```

## Development Timeline (M.Tech Thesis)
This framework is being actively developed over a strict 11-week schedule. Please refer to `PROJECT_MILESTONES.md` for a highly detailed breakdown of completed milestones and upcoming modules.
