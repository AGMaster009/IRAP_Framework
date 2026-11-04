# IRAP Framework: Technical Q&A and Strategy

This document answers core technical and strategic questions regarding the execution of the IRAP Framework, specifically tailored for medical dataset processing and M.Tech thesis documentation.

---

## 1. Datasets: What are we using, how to get them, and what have we used so far?

### What we have used so far:
Currently, the codebase only utilizes a tiny `sample.csv` (inside the `datasets/` folder) which was manually created to unit-test the data loader and feature profilers. We have not run the framework on a real medical dataset yet.

### Datasets we *will* use (and how to get them):
1.  **Public Benchmark Datasets (Required for Academic Baselines):**
    *   *Where to get them:* The [UC Irvine (UCI) Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) and [Kaggle](https://www.kaggle.com/).
    *   *Specifics:* You should download the **Heart Disease**, **Pima Indians Diabetes**, and **Breast Cancer Wisconsin** datasets. These are globally recognized standard datasets in medical ML. Proving IRAP works on these makes your thesis irrefutable.
2.  **Live Hospital Data:**
    *   This is your proprietary, novel dataset that you are currently procuring from the hospital.
3.  **Synthetic (Mathematically Generated) Datasets:**
    *   *Where to get them:* We will write a Python script in Week 10 to mathematically generate them using `numpy` (e.g., generating columns where `Blood_Pressure = Age * 1.5 + log(BMI)`). This is required to scientifically prove our algorithm can find hidden equations.

---

## 2. Entire Project Flow (Technical Top-to-Bottom)

Here is the exact technical pipeline the framework executes, step-by-step:

1.  **Data Ingestion (`DatasetLoader`)**: Loads CSV/Excel files using `chardet` for strict encoding fallback and `csv.Sniffer` for strict delimiter detection, avoiding data corruption. Returns a central `Dataset` object.
2.  **Dataset Profiling (`DatasetProfiler`)**: Calculates global architecture metrics: total memory footprint, row/col dimensions, global missing percentages, duplicate rows, and target class imbalances.
3.  **Feature Profiling (`FeatureProfiler`)**: Loops through every column independently.
    *   *Numeric:* Calculates Variance, Skewness, Kurtosis, Interquartile Range (IQR) for outliers, and executes **D'Agostino's K-squared test** to prove if the distribution is Gaussian (Normal).
    *   *Categorical:* Calculates Mode, Cardinality, and **Shannon Entropy** to measure information gain.
4.  **Relationship Discovery (Upcoming)**: Calculates linear relationships (Mutual Information/Pearson) and Hidden Relationships (SHAP) between columns.
5.  **Context Knowledge Graph (Upcoming)**: Uses the `NetworkX` library to build a mathematical directed graph where Nodes = Features, and Edges = Relationship Strength.
6.  **Intelligent Decision Engine (Upcoming)**: A deterministic engine that reads the Profiles and the Graph to output a specific preprocessing plan. For example: *If Skewness > 1 ➔ Recommend RobustScaler.* OR *If Missing > 20% and Feature A depends heavily on B and C ➔ Recommend Iterative KNN Imputation using B and C.*
7.  **Pipeline Validation (Upcoming)**: Evaluates if a Random Forest model trained on the IRAP-processed data outperforms a Random Forest trained on data processed by standard Scikit-Learn `SimpleImputer`.

---

## 3. Identifying Hidden Relationships (The Process Flow)

Traditional statistics (like Pearson correlation) can only see straight lines. If age causes blood pressure to rise, but then drop after age 80, Pearson misses it. Here is the technical flow to find those hidden, non-linear dependencies:

1.  **Iterative Target Assignment:** We loop through every feature in the dataset (e.g., Feature A). We temporarily treat Feature A as the "Target Variable".
2.  **Surrogate Modeling:** We train a highly non-linear tree-based model (like a **Random Forest** or **XGBoost** regressor/classifier) using all the *other* features (B, C, D) to predict Feature A.
3.  **SHAP TreeExplainer:** We pass that trained Random Forest into the `shap.TreeExplainer` algorithm.
4.  **Interaction Value Extraction:** SHAP calculates exact marginal contributions based on game theory. We extract the **SHAP Interaction Values**, which explicitly measure if the effect of Feature B on Feature A changes depending on the value of Feature C.
5.  **Graph Construction:** If the SHAP interaction value is exceptionally high, we programmatically draw a weighted edge between A, B, and C in our NetworkX Knowledge Graph.

---

## 4. Standardized Medical Units & Thresholds (Where to find them)

If you are preparing a custom dataset from live hospital records, you must map the raw columns to globally standardized thresholds. Do not invent thresholds; use the following official medical bodies:

### Where to get the standardized definitions:
1.  **LOINC (Logical Observation Identifiers Names and Codes):** This is the international standard for identifying health measurements, observations, and documents. If your hospital gives you a column named `Bld_Sug_F`, you should map it to its official LOINC code for "Fasting Glucose". You can search their database at [loinc.org](https://loinc.org/).
2.  **WHO (World Health Organization) Guidelines:** Excellent for BMI, weight, and general health thresholds.
3.  **Specialized Medical Associations:**
    *   **Diabetes (Blood Sugar):** American Diabetes Association (ADA).
    *   **Heart/Blood Pressure:** American Heart Association (AHA).
    *   **Cholesterol/Lipids:** National Cholesterol Education Program (NCEP).

### Example: Fasting Blood Sugar (Fasting Plasma Glucose) via ADA Guidelines
If you are generating or cleaning a dataset with fasting blood sugar, the universal ADA thresholds (often measured in mg/dL) are:
*   **Normal:** Less than 100 mg/dL (5.6 mmol/L)
*   **Prediabetes (Impaired Fasting Glucose):** 100 mg/dL to 125 mg/dL (5.6 to 6.9 mmol/L)
*   **Diabetes:** 126 mg/dL (7.0 mmol/L) or higher on two separate tests.

### How, where, and why do we use LOINC and these standard thresholds?
*   **Why:** Real-world hospital data is chaotic. One doctor might label a column "Fasting_Glucose", another "Bld_Sug_F", and another "FBS". LOINC provides a universal, standardized ID (e.g., LOINC 14771-0). We use it for **Semantic Interoperability**—so the framework knows exactly what biological metric it is analyzing regardless of the hospital's naming convention.
*   **Where:** This is used during the **Dataset Profiling / Ingestion phase**. We map the raw column names to their LOINC codes to attach clinical metadata to the `FeatureProfile`.
*   **How:** We use these standards as **Biological Guardrails**. For example, if the framework knows a column is LOINC 14771-0 (Fasting Glucose in mg/dL), it can instantly apply the ADA thresholds. If it sees a value of `9000`, it doesn't need machine learning to know it's an outlier; it uses the standard threshold to instantly flag it as a sensor error or typo.

### Do we train the machine learning model using this standard data?
**Absolutely not.** 
The machine learning models (Random Forest, SHAP) are trained strictly and only on the *actual patient data* to discover the true, hidden relationships. 
The LOINC codes and WHO/ADA thresholds are never fed into the ML model as training data. Instead, they act as a **Rules Engine / Safeguard** for the pipeline. For example, if the SHAP-based imputation engine accidentally tries to impute a missing patient age as `300` or a blood sugar as `-5` (which algorithms sometimes do mathematically), the framework uses the standardized medical definitions to block that imputation, recognizing it is biologically impossible.

---

## 5. SHAP: Why is it the Core Algorithm and What are the Alternatives?

Yes, SHAP (SHapley Additive exPlanations) is the core algorithm for the *novel* part of this thesis (identifying complex hidden relationships). Here is exactly how to defend this choice:

### Why SHAP?
SHAP is rooted in cooperative game theory (Lloyd Shapley won a Nobel Memorial Prize for the underlying math). It treats every feature as a "player" in a game and calculates exactly how much each player contributed to the final prediction. Crucially, the `shap.TreeExplainer` algorithm explicitly calculates **SHAP Interaction Values**—a matrix showing exactly how Feature A and Feature B *worked together* to influence Feature C. This matrix translates perfectly into our Context Knowledge Graph (Nodes and Edges).

### Are there better alternatives? What did we consider?
We considered several alternatives, but SHAP is theoretically superior for our specific use case:

1.  **LIME (Local Interpretable Model-agnostic Explanations):**
    *   *Why we rejected it:* LIME builds local, linear approximations around single predictions. It is not globally consistent, and it does not naturally produce a unified, global pairwise interaction matrix that we can turn into a Knowledge Graph.
2.  **Permutation Feature Importance (PFI):**
    *   *Why we rejected it:* PFI calculates importance by randomly shuffling a column and seeing how much the model accuracy drops. However, in medical data with highly correlated features (like BMI and Weight), shuffling one feature creates physically impossible synthetic data points (e.g., a person with high BMI but extremely low weight). PFI evaluates the model on nonsense data, ruining the relationship scores. SHAP relies on conditional expectations, avoiding this entirely.
3.  **DeepLIFT or Integrated Gradients:**
    *   *Why we rejected it:* These are mathematically excellent, but they *only* work on Deep Neural Networks. Tabular medical data heavily favors Tree-based algorithms (Random Forests, XGBoost). Therefore, we chose Tree-based surrogate models, and for trees, **TreeSHAP** is mathematically exact and heavily optimized.
4.  **Partial Dependence Plots (PDP) / ICE:**
    *   *Why we rejected it:* Great for human visualization, but computationally impossible to extract a programmatic multi-dimensional interaction matrix from them efficiently.

By measuring the final downstream model accuracy of all three, your thesis will contain a quantitative chart proving exactly *why* SHAP was the optimal choice.

---

## 6. Deep Dive: SHAP's Mechanics and Thesis Novelty

If the panel presses you hard on the specific use of SHAP, here is exactly how to answer them:

**1. How does SHAP actually work?**
SHAP is based on cooperative game theory (developed by Lloyd Shapley). Imagine the features are "players" cooperating to win a "game" (the model's prediction). SHAP calculates the *marginal contribution* of each player by testing every possible combination (coalition) of players. For our framework, we use **TreeSHAP**, which mathematically exploits the structure of Decision Trees/Random Forests to compute these exact contributions efficiently in polynomial time, rather than exponential time.

**2. Why is SHAP perfect for this scenario?**
Because to build a Context Knowledge Graph, we need precise, mathematically sound "weights" for our edges. SHAP is the only method that satisfies the property of *Additivity* (the sum of feature impacts equals the total prediction) and *Consistency*. Crucially, it provides an **Interaction Matrix**—a literal mathematical grid showing exactly how much Feature A depends on Feature B. This matrix maps directly to the Nodes and Edges of a Knowledge Graph.

**3. Are there any alternatives for the exact same kind of interaction extraction?**
Beyond LIME or PFI, we could theoretically use **Friedman's H-statistic** or **Greenwell's Interaction Index**. However, the H-statistic is computationally brutal for high-dimensional data and doesn't distribute the interaction effect linearly among features. Alternatively, we could use **Attention Weights** from Transformer Neural Networks (like TabNet), but deep learning requires massive datasets to converge and often overfits on the smaller tabular datasets typical in medical research. TreeSHAP provides the perfect balance of exact math and tabular data compatibility.

**4. Isn't SHAP for explainability? Why does it work here?**
*This is the most important question to answer correctly.*
Yes, SHAP is an Explainable AI (XAI) tool. Traditionally, data scientists use it at the very *end* of the pipeline to explain a final model's predictions to stakeholders. 
**We are inverting this paradigm.** We are using SHAP at the very *beginning* (the preprocessing phase). We train temporary "surrogate models" on the raw data, use SHAP's explainability to map the hidden topology of the dataset, and then throw the surrogate model away. We use the XAI output to drive preprocessing decisions.

**5. Isn't SHAP already used everywhere? How is that a novelty?**
Yes, SHAP is ubiquitous for *post-hoc model explainability* (e.g., explaining why a specific patient was diagnosed with diabetes). 
**The Novelty:** The novelty of the IRAP framework is **not** inventing SHAP. The novelty is **repurposing XAI (SHAP) for dynamic Data Preprocessing and Imputation**. 
Current literature rarely uses SHAP to build a Knowledge Graph that dynamically dictates whether a missing value should be imputed via Mean vs. KNN, or whether a column should be One-Hot vs. Target Encoded. Moving XAI from the *end* of the pipeline (explaining a model) to the *start* of the pipeline (intelligently preparing the data) is a significant and highly defensible academic contribution.

---

## 7. If LOINC is Just Guardrails, Then How Does IRAP Actually Decide What Value to Impute?

This is a critical distinction. The LOINC standards only tell the framework **what is biologically impossible**. The actual intelligence of *what value to fill in* comes entirely from the patient data itself. Here is the exact two-layer process:

**Layer 1: The SHAP Knowledge Graph tells IRAP *how* to impute.**
Suppose Patient #42 has a missing value in the `Blood_Glucose` column. IRAP does not blindly use the average. Instead, it consults the Context Knowledge Graph. If the graph says `Blood_Glucose` has strong SHAP interactions with `BMI` and `Insulin`, the Decision Engine dynamically builds a KNN or Iterative Imputation model specifically using those two features as predictors. The missing `Blood_Glucose` is then estimated from the patient's actual `BMI` and `Insulin` values.

**Layer 2: The LOINC / ADA standard acts as the final validation check.**
After Layer 1 computes the imputed value (e.g., `95 mg/dL`), the guardrail checks: is this within the biological range of a living human? If the algorithm had instead computed `-50 mg/dL` (which can happen with edge-case KNN), the guardrail rejects it and falls back to the column median. This prevents mathematically absurd results from corrupting the dataset.

**In Summary:**
> SHAP + the patient data = *What* the value is estimated to be.
> LOINC + Medical Standards = *Validating* the result is biologically possible.

---

## 8. Why Do We Need the Heart Disease, Pima Indians Diabetes, and Breast Cancer Wisconsin Datasets?

These three datasets serve one specific purpose in the thesis: **proving that IRAP is objectively better than standard preprocessing**. Here is exactly how:

**The Experiment (Week 10 - Ablation Study):**
1. Take the Pima Indians Diabetes dataset (which has significant missing values, outliers, and class imbalance — a perfect test case).
2. **Baseline Pipeline:** Run it through a standard Scikit-Learn pipeline (`SimpleImputer(strategy='mean')` ➔ `StandardScaler()` ➔ `RandomForestClassifier`). Record the AUC, F1-Score, and Accuracy.
3. **IRAP Pipeline:** Run the same raw dataset through IRAP (context-aware KNN imputation ➔ relationship-aware scaling ➔ `RandomForestClassifier`). Record the AUC, F1-Score, and Accuracy.
4. **Prove the gap:** If IRAP scores higher (which the theory strongly predicts), you have a quantitative, empirical, side-by-side proof that your framework produces cleaner data and better models.

**Why these three specifically?**
*   They are **universally recognized** in academic medical ML literature. Every reviewer, panel member, and your guide will already be familiar with them. Using a random obscure dataset would raise questions; using these specific datasets gives your comparison immediate credibility.
*   They each have **different types of preprocessing challenges**: Pima Indians has missing values, Heart Disease has mixed data types, Breast Cancer Wisconsin has outlier sensitivity. Using all three proves IRAP generalizes across different problems.
*   They give you **reproducibility**: Any other researcher in the world can download the exact same datasets and reproduce your results to verify your claims.

---

## 9. What are some examples of "Traditional" Preprocessing Algorithms, and why are they flawed?

If the panel asks for examples of the "traditional" algorithms that IRAP is designed to replace or improve upon, categorize them like this:

**1. Traditional Missing Value Imputation (Univariate methods):**
*   **Mean / Median / Mode Imputation:** (e.g., Scikit-Learn's `SimpleImputer`). This replaces all missing values in a column with the column's average. 
    *   *The Flaw:* It destroys variance and completely ignores feature relationships. If a 20-year-old and an 80-year-old are both missing blood pressure, it gives them both the exact same average blood pressure.
*   **Listwise Deletion (Dropna):** Dropping any row that has a missing value. 
    *   *The Flaw:* In medical datasets, you lose massive amounts of critical patient data.

**2. Traditional Categorical Encoding:**
*   **One-Hot Encoding (Dummy Variables):** Creates a new binary column (0 or 1) for every single unique category.
    *   *The Flaw:* If a medical code column has 500 unique disease codes, One-Hot Encoding creates 500 new columns. This causes the "Curse of Dimensionality" and creates massive, sparse, memory-heavy datasets that cause ML models to overfit.
*   **Label Encoding:** Converts categories to arbitrary numbers (e.g., Asthma=1, Diabetes=2, Flu=3).
    *   *The Flaw:* ML models mistakenly think these arbitrary numbers have mathematical meaning (e.g., that Flu is mathematically "greater" than Asthma).

**3. Traditional Scaling / Normalization:**
*   **StandardScaler (Z-score normalization) or Min-Max Scaler:** 
    *   *The Flaw:* These standard scalers apply a blanket mathematical formula to the entire column. If a column has massive medical outliers (e.g., a sensor error reading a heart rate of 500), these traditional scalers get heavily distorted by the outlier, ruining the scale for all the healthy patients.

**The IRAP Defense:**
*"Traditional preprocessing algorithms are 'blind' and 'univariate'. They apply simple math to a single column in isolation without looking at the rest of the dataset. IRAP is 'multivariate' and 'relationship-aware'—it looks at the entire dataset's topology to make intelligent, context-specific decisions (e.g., using Target Encoding for high-cardinality data, or K-Nearest Neighbors for imputation based on related features)."*

---

## 10. Do Linear Regression, Logistic Regression, and Pearson's Coefficient count as Preprocessing or Value Identification algorithms?

This is a great trick question a panel might ask to test your fundamental understanding of machine learning pipelines. Here is exactly how to categorize them:

**1. Pearson's Correlation Coefficient: YES (Value Identification / Preprocessing)**
Pearson's is strictly a statistical tool used during the Exploratory Data Analysis (EDA) and preprocessing phase. It is used to identify relationships between variables.
*   *Why IRAP doesn't rely solely on it:* Pearson's coefficient is fundamentally limited because it only measures **linear** relationships between exactly two variables. If a patient's health declines steeply after age 60 (a non-linear curve), Pearson's completely misses it. This is why IRAP uses SHAP to find non-linear, complex multi-variable interactions instead.

**2. Linear & Logistic Regression: IT DEPENDS (Traditionally NO, but structurally YES)**
*   **Traditionally:** No. Linear Regression (for continuous numbers) and Logistic Regression (for binary classification) are considered **Predictive ML Models**. They sit at the very *end* of the pipeline. You use them to predict the final outcome (like predicting if a patient has cancer).
*   **In Advanced Preprocessing (Like IRAP or MICE): YES.**
    In advanced data science, we take these predictive models and drag them backward into the preprocessing phase to act as **Surrogate Models** for imputation. For example, the famous `MICE` (Multiple Imputation by Chained Equations) algorithm literally runs a mini Linear/Logistic Regression on the dataset just to predict and fill in a missing value. 

**Summary for the Panel:**
*"Pearson's coefficient is a traditional relationship-discovery tool in preprocessing, but it is limited to straight, linear lines. Linear and Logistic Regression are traditionally final predictive models, but in advanced frameworks, they are often repurposed as 'surrogate models' during the preprocessing phase to intelligently predict and impute missing values."*

---

## 11. Are there any Evolutionary Algorithms (e.g., Genetic Algorithms, Particle Swarm) used in IRAP? Will they be used in the future?

**No, they are not currently used, and they are intentionally excluded from the core IRAP pipeline.**

If the panel asks about evolutionary or "nature-inspired metaheuristic" algorithms, here is how you defend your architectural choice:

**Why IRAP avoids Evolutionary Algorithms:**
1.  **They lack deterministic explainability:** Evolutionary algorithms (like Genetic Algorithms or Ant Colony Optimization) use stochastic (randomized) processes and mutations to search for an optimal imputed value. They operate as a "black box." In the medical domain, doctors and regulators need to know *exactly* why a specific value was imputed. IRAP uses SHAP and deterministic rules to provide a clear, trackable, mathematically exact audit trail.
2.  **Computational Brutality:** Evolutionary algorithms require hundreds of generations/iterations to converge, making them incredibly slow and computationally expensive for large, high-dimensional datasets. IRAP's use of TreeSHAP is highly optimized and runs in polynomial time, making it much faster for tabular data.
3.  **Risk of Overfitting:** Evolutionary algorithms can easily overfit to the training data, creating imputed values that look perfect mathematically but are biologically absurd.

**Future Scope (Where they *could* be used):**
While they will not be used for the core imputation logic, you can propose them as a "Future Work" item for **Hyperparameter Tuning**. For instance, in the future, a Genetic Algorithm could be wrapped *around* the IRAP framework to automatically search for the optimal threshold values (e.g., finding the exact SHAP threshold cut-off or the optimal `k` for the KNN imputer) to maximize the downstream model's accuracy.

---

## 12. Biological Guardrails: How were they built from LOINC, and why are they robust without hardcoded numerical bounds?

If the panel asks how you built the biological guardrails, especially considering that the raw LOINC datasets do not contain explicit numerical ranges (e.g., they don't explicitly state "Heart Rate must be between 40 and 200"), here is your definitive, highly academic defense.

### How we achieved the result (The Technical Steps):
1.  **Massive Data Consolidation:** We ingested 31 distinct LOINC medical datasets (spanning Pathology, EKG, Chemistry, etc.) representing over **91,600** raw medical terminology codes.
2.  **Deduplication & Extraction:** We parsed these files into a centralized Pandas DataFrame, dropping duplicates based on the medical `COMPONENT` (the feature name) to create a unique lookup dictionary of over **40,000** medical tests. From these, we strictly extracted the `SCALE_TYP` (Quantitative vs Nominal) and `EXAMPLE_UNITS` (e.g., `kg`, `%`, `mm Hg`).
3.  **Heuristic Physics Engine:** Because LOINC doesn't provide hard numerical bounds, we built a Python engine (`guardrail_builder.py`) to procedurally generate them based on the laws of physics and medical scale integrity. The script scans the units and dynamically attaches hard boundaries to the data.
4.  **O(1) JSON Lookup:** We exported the finalized, rules-based dictionary into a highly optimized JSON file (`biological_guardrails.json`) that the Week 6 Imputation module queries instantly.

### Why did we take these steps? (The Academic Defense):

Building guardrails based on Scale and Units is academically far more robust than manually hardcoding numbers, for three critical reasons:

**1. Enforcing "Scale Integrity" (Solving a major flaw in basic ML)**
Standard algorithms (like a naive KNN Imputer) are mathematically "blind". If a hospital column has missing values for a categorical test like "Blood Type", a standard KNN might average its neighbors and impute a continuous decimal value like `1.5`. 
By using the LOINC `SCALE_TYP` column, IRAP knows definitively whether a feature is Quantitative (`Qn`) or Nominal (`Nom`). This gives us a hard guardrail: **IRAP will actively block continuous decimal imputations on Nominal medical features.**

**2. Procedural "Heuristic Physics" Boundaries**
While the dataset doesn't explicitly state "Heart rate must be between 40 and 150," the `EXAMPLE_UNITS` column gives us the laws of physics. Our engine maps units to physical realities:
*   Any unit containing `%` is automatically given a hard, mathematically unbreakable guardrail of `[0, 100]`.
*   Any unit representing Mass, Volume, Count, or Pressure (`kg`, `mg/dL`, `mm Hg`, `mL`) is automatically given a hard guardrail of `minimum: 0.0`. This mathematically prevents the imputation algorithms from ever generating a negative body weight, a negative cell count, or a negative blood pressure.

**3. Utilizing Global Standards (Interoperability)**
In a thesis defense, if you manually typed out bounds like "Blood Pressure max is 250", the panel could argue that your bounds are arbitrary. By utilizing LOINC, you are leveraging an international, WHO-recognized standard for health measurements. 
You can confidently state: *"The IRAP framework integrates the official, internationally recognized LOINC nomenclature database, parsing over 40,000 clinical terms to dynamically and procedurally enforce data type integrity and physical boundaries."*

This approach transforms IRAP from a simple ML script into a rigorous, medically-aware informatics framework.

**4. How do we validate the script, and what if the heuristic data is wrong?**
A panel member might challenge the safety of the `guardrail_builder.py` script: *What if the script makes a mistake and applies a minimum of 0 to a biological metric that CAN be negative (like Temperature in Celsius)?* 
Here is how you defend the system's safety and validation:

*   **Conservative Fail-Safe Design:** The script was engineered to be extremely conservative. It only applies boundaries to strings that are mathematically absolute (e.g., `%`, `kg`, `mm Hg`). It contains explicit exclusion logic (e.g., bypassing anything containing `temp` or `celsius`). If the script is ever unsure, it defaults to `min_value: null`, gracefully deferring to the ML imputer rather than risking a false guardrail.
*   **Deterministic Transparency:** Unlike a neural network, the output of the script is a highly readable JSON file. It is not a "black box." A researcher or domain expert can manually open `biological_guardrails.json` and verify the exact bounds assigned to their specific dataset's columns in seconds.
*   **Architectural Overridability:** The JSON file acts as an external configuration, not hard-coded Python logic. The script merely serves as an *accelerator* to build the massive baseline dictionary. If a doctor identifies an edge case where a guardrail is too strict, they can manually edit the JSON file to override the heuristic instantly.
