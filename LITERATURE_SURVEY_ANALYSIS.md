# Literature Survey Analysis: Advanced Imputation Methods

This document analyzes the 6 provided research papers, summarizes their core approaches, and explicitly maps their relevance to our Intelligent Relationship-Aware Preprocessing (IRAP) Framework.

---

### 1. CMI: Cluster-Centric Missing Value Imputation with Feature Consistency
*   **Core Approach:** Uses K-Means clustering to segment data and imputes missing values based on cluster centroids. It uniquely uses **SHAP values** *post-imputation* to verify that the imputed dataset maintains the same feature importance ranking as medical baselines.
*   **Relevance to IRAP:** **Extremely High**
*   **Justification:** This paper validates our core thesis that preserving feature relationships is critical. However, they use SHAP *after* imputation to check their work. IRAP is fundamentally more advanced because we use SHAP *before* imputation to construct the Knowledge Graph that actively decides the imputation strategy. You can cite this paper to prove SHAP is academically recognized for feature consistency.

### 2. Attention-based Imputation of Missing Values in Electronic Health Records Tabular Data
*   **Core Approach:** Adapts Transformer architectures (Self-Attention) and contrastive learning (CutMix, MixUp) to learn between-feature and between-sample attention weights to reconstruct missing tabular EHR data.
*   **Relevance to IRAP:** **High**
*   **Justification:** This paper shares our motivation: capturing complex feature interactions (attention) is better than isolated statistics. While they use deep learning Transformers and we use TreeSHAP + Knowledge Graphs, you can cite this to defend the necessity of "relationship-aware" (attention/interaction) mechanisms specifically for Electronic Health Records.

### 3. Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity
*   **Core Approach:** Proposes a Federated Learning (FL) approach to impute data across multiple hospitals without sharing raw data. It specifically addresses complex Missing Not At Random (MNAR) mechanisms.
*   **Relevance to IRAP:** **Medium**
*   **Justification:** IRAP is currently a centralized framework, not federated. However, this paper is crucial for your **Future Scope / Semester 3** section. You can propose that extending IRAP into a "Federated IRAP" (allowing multiple hospitals to collaboratively build the SHAP Knowledge Graph without sharing patient data) is the next logical evolution of your thesis.

### 4. Generative Adversarial Networks Assist Missing Data Imputation: A Comprehensive Survey
*   **Core Approach:** A systematic review of using GANs (like GAIN and MisGAN) to generate synthetic samples and impute missing data, concluding that GANs outperform traditional ML in complex scenarios.
*   **Relevance to IRAP:** **Medium-High**
*   **Justification:** This paper serves as the perfect baseline comparison. Your panel might ask, "Why not use GANs?" You can cite this survey, acknowledging GANs are powerful, but argue that GANs operate as "black boxes." IRAP's use of SHAP provides *deterministic explainability* (via the Knowledge Graph), which is legally and clinically preferred in the medical industry over black-box GANs.

### 5. Systematic Review of Using Machine Learning in Imputing Missing Values
*   **Core Approach:** A massive review of 94 papers (2016-2021) showing that "Hybrid Models" and Deep Learning have overtaken traditional statistical imputers. It highlights that most studies ignore MNAR mechanisms and lack explainability.
*   **Relevance to IRAP:** **High**
*   **Justification:** This is your "Research Gap" goldmine. You can cite this paper to state: *"Recent systematic reviews show that while ML imputation is trending, the literature severely lacks explainable, transparent methods."* IRAP directly fills this gap by being a hybrid, highly explainable framework.

### 6. Missing Value Imputation Designs and Methods of Nature-Inspired Metaheuristic Techniques
*   **Core Approach:** Reviews the use of optimization algorithms (like Particle Swarm Optimization, Genetic Algorithms) to find the absolute mathematically optimal imputed value by minimizing complex error functions.
*   **Relevance to IRAP:** **Low-Medium**
*   **Justification:** While fascinating, metaheuristic optimization is computationally brutal and prone to overfitting. You can cite this paper to defend why IRAP uses deterministic decision engines based on SHAP graphs rather than stochastic evolutionary searches, prioritizing clinical explainability over black-box optimization.

---

### Conclusion for your Thesis:
All 6 papers are highly relevant to your domain and strictly support the narrative that standard imputation is obsolete. They provide the perfect academic backdrop to justify why your specific approach (SHAP + Knowledge Graphs) is novel, necessary, and clinically superior.
