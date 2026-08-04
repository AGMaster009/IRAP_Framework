"""
feature_profile.py
------------------

Represents a single feature (column) within a dataset.

Each column in the dataset has one corresponding FeatureProfile
object that accumulates information throughout the IRAP pipeline.

The FeatureProfile evolves as the framework progresses:

Dataset Loader
        ↓
Dataset Profiler
        ↓
Relationship Discovery
        ↓
Hidden Relationship Discovery
        ↓
Decision Engine
        ↓
Preprocessing
        ↓
Validation
        ↓
Explainability
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureProfile:

    # ==========================================================
    # BASIC INFORMATION
    # ==========================================================

    feature_name: str = ""

    display_name: str = ""

    data_type: str = ""

    column_index: int = -1

    description: str = ""

    # ==========================================================
    # BASIC STATISTICS
    # ==========================================================

    mean: Optional[float] = None

    median: Optional[float] = None

    mode: Optional[Any] = None

    minimum: Optional[float] = None

    maximum: Optional[float] = None

    variance: Optional[float] = None

    standard_deviation: Optional[float] = None

    range_value: Optional[float] = None

    interquartile_range: Optional[float] = None

    entropy: Optional[float] = None

    skewness: Optional[float] = None

    kurtosis: Optional[float] = None

    # ==========================================================
    # VALUE INFORMATION
    # ==========================================================

    unique_values: int = 0

    unique_percentage: float = 0.0

    cardinality: str = "Unknown"

    constant_feature: bool = False

    # ==========================================================
    # MISSING VALUES
    # ==========================================================

    missing_count: int = 0

    missing_percentage: float = 0.0

    has_missing_values: bool = False

    # ==========================================================
    # DISTRIBUTION
    # ==========================================================

    distribution_type: str = "Unknown"

    normality_score: Optional[float] = None

    distribution_notes: str = ""

    # ==========================================================
    # OUTLIERS
    # ==========================================================

    outlier_count: int = 0

    outlier_percentage: float = 0.0

    has_outliers: bool = False

    outlier_detection_method: str = ""

    # ==========================================================
    # RELATIONSHIP DISCOVERY
    # ==========================================================

    related_features: List[str] = field(default_factory=list)

    correlation_scores: Dict[str, float] = field(default_factory=dict)

    mutual_information_scores: Dict[str, float] = field(default_factory=dict)

    relationship_strength: Dict[str, str] = field(default_factory=dict)

    # ==========================================================
    # HIDDEN RELATIONSHIPS
    # ==========================================================

    interaction_features: List[str] = field(default_factory=list)

    hidden_dependencies: List[str] = field(default_factory=list)

    feature_importance: Optional[float] = None

    importance_rank: Optional[int] = None

    shap_importance: Optional[float] = None

    # ==========================================================
    # DECISION ENGINE
    # ==========================================================

    recommended_imputation: str = ""

    recommended_encoding: str = ""

    recommended_scaling: str = ""

    recommended_transformation: str = ""

    recommended_outlier_method: str = ""

    preprocessing_priority: int = 0

    # ==========================================================
    # VALIDATION
    # ==========================================================

    validation_rules: List[str] = field(default_factory=list)

    confidence_score: Optional[float] = None

    passed_validation: bool = False

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================

    contributing_features: List[str] = field(default_factory=list)

    explanation: str = ""

    reasoning: str = ""

    # ==========================================================
    # WARNINGS & RECOMMENDATIONS
    # ==========================================================

    warnings: List[str] = field(default_factory=list)

    recommendations: List[str] = field(default_factory=list)

    # ==========================================================
    # TIMESTAMPS
    # ==========================================================

    created_at: datetime = field(default_factory=datetime.now)

    updated_at: datetime = field(default_factory=datetime.now)

    # ==========================================================
    # HELPER METHODS
    # ==========================================================

    def update_timestamp(self) -> None:
        """
        Updates the modification timestamp.
        """
        self.updated_at = datetime.now()

    def add_related_feature(
        self,
        feature_name: str,
        correlation: Optional[float] = None
    ) -> None:
        """
        Adds a related feature discovered during
        relationship analysis.
        """

        if feature_name not in self.related_features:
            self.related_features.append(feature_name)

        if correlation is not None:
            self.correlation_scores[feature_name] = correlation

        self.update_timestamp()

    def add_warning(self, warning: str) -> None:
        """
        Adds a warning generated during profiling.
        """

        self.warnings.append(warning)

        self.update_timestamp()

    def add_recommendation(self, recommendation: str) -> None:
        """
        Adds a preprocessing recommendation.
        """

        self.recommendations.append(recommendation)

        self.update_timestamp()

    def is_numeric(self) -> bool:
        """
        Returns True if the feature is numeric.
        """

        return self.data_type.lower() in [
            "int",
            "int64",
            "float",
            "float64",
            "numeric"
        ]

    def is_categorical(self) -> bool:
        """
        Returns True if the feature is categorical.
        """

        return self.data_type.lower() in [
            "object",
            "category",
            "string"
        ]

    def is_boolean(self) -> bool:
        """
        Returns True if the feature is boolean.
        """

        return self.data_type.lower() in [
            "bool",
            "boolean"
        ]

    def is_datetime(self) -> bool:
        """
        Returns True if the feature is datetime.
        """

        return "datetime" in self.data_type.lower()

    def summary(self) -> Dict[str, Any]:
        """
        Returns a lightweight summary of the feature.
        """

        return {
            "feature_name": self.feature_name,
            "data_type": self.data_type,
            "missing_percentage": self.missing_percentage,
            "outliers": self.outlier_count,
            "related_features": len(self.related_features),
            "recommendation": self.recommended_imputation,
            "confidence": self.confidence_score
        }

    def __repr__(self) -> str:

        return (
            f"FeatureProfile("
            f"feature='{self.feature_name}', "
            f"type='{self.data_type}', "
            f"missing={self.missing_percentage:.2f}%, "
            f"outliers={self.outlier_count})"
        )