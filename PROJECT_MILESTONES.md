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

## 🟡 Week 2: Feature Profiling Module
**Dates:** Aug 8 – Aug 15  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 3: Relationship Discovery Module
**Dates:** Aug 16 – Aug 23  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 4: Hidden Relationship Discovery + Knowledge Graph
**Dates:** Aug 24 – Aug 31  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 5: Intelligent Decision Engine
**Dates:** Sept 1 – Sept 7  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 6: Context-Aware Missing Value Processing
**Dates:** Sept 8 – Sept 15  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 7: Outlier, Encoding, Scaling Modules
**Dates:** Sept 16 – Sept 23  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 8: Validation Module
**Dates:** Sept 24 – Sept 30  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 9: Explainability Module
**Dates:** Oct 1 – Oct 10  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 10: Experiments on Multiple Datasets
**Dates:** Oct 11 – Oct 20  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Week 11: Thesis Writing, Graphs, Final Testing
**Dates:** Oct 21 – Oct 31  
**Status:** Not Started  
*Documentation will be added upon completion.*

---

## 🔴 Buffer: Bug Fixes, Review Corrections, Viva Preparation
**Dates:** Nov 1 – Nov 7  
**Status:** Not Started  
*Documentation will be added upon completion.*
