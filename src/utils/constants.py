"""
constants.py
------------
Global constants, default thresholds, and configuration settings
for the IRAP Framework.
"""

# Profiling & Health Thresholds
MISSING_DROP_THRESHOLD = 50.0          # Drop column if missing > 50%
HIGH_CARDINALITY_THRESHOLD = 15        # Route to Target/Frequency Encoding if categories > 15
HIGH_DIMENSIONALITY_THRESHOLD = 100    # Warn if columns > 100
HIGH_MISSING_DATASET_THRESHOLD = 20.0  # Dataset health alert if overall missing > 20%
IMBALANCE_RATIO_THRESHOLD = 3.0        # Alert if majority class / minority class > 3:1

# Relationship Discovery Thresholds
RELATIONSHIP_SHAP_THRESHOLD = 0.10     # Minimum normalized SHAP interaction to form Knowledge Graph edge
MAX_SURROGATE_SAMPLES = 5000           # Sample limit to guarantee zero Out-Of-Memory crashes

# Statistical Thresholds
NORMALITY_ALPHA = 0.05                 # D'Agostino K-squared test alpha level
SKEWNESS_THRESHOLD = 1.0               # Skewness beyond [-1.0, 1.0] triggers RobustScaler
OUTLIER_IQR_FACTOR = 1.5               # Tukey's fence IQR multiplier

# Biological Guardrails Default Path
DEFAULT_GUARDRAILS_PATH = "datasets/biological_guardrails.json"

# Supported File Extensions
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

