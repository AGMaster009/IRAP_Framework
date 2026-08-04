"""
dataset_profile.py
------------------

Stores high-level information about an entire dataset.

This class is populated by the DatasetProfiler module after
the dataset has been successfully loaded.

It contains only dataset-level information and does NOT store
individual feature statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DatasetProfile:
    """
    Represents the overall profile of a dataset.

    This class stores high-level metadata and statistics that
    describe the dataset as a whole.
    """

    # ==========================================================
    # BASIC INFORMATION
    # ==========================================================

    dataset_name: str = ""

    file_name: str = ""

    file_type: str = ""

    file_size_mb: float = 0.0

    # ==========================================================
    # DIMENSIONS
    # ==========================================================

    num_rows: int = 0

    num_columns: int = 0

    memory_usage_mb: float = 0.0

    # ==========================================================
    # COLUMN INFORMATION
    # ==========================================================

    numeric_columns: int = 0

    categorical_columns: int = 0

    boolean_columns: int = 0

    datetime_columns: int = 0

    text_columns: int = 0

    # ==========================================================
    # DATA QUALITY
    # ==========================================================

    total_missing_values: int = 0

    missing_percentage: float = 0.0

    duplicate_rows: int = 0

    duplicate_percentage: float = 0.0

    total_outliers: int = 0

    quality_score: Optional[float] = None

    # ==========================================================
    # TARGET INFORMATION
    # ==========================================================

    target_column: Optional[str] = None

    target_type: Optional[str] = None

    # ==========================================================
    # DATASET SUMMARY
    # ==========================================================

    class_distribution: Dict[str, Any] = field(default_factory=dict)

    data_types: Dict[str, str] = field(default_factory=dict)

    missing_by_column: Dict[str, int] = field(default_factory=dict)

    unique_values: Dict[str, int] = field(default_factory=dict)

    
    # ==========================================================
    # DATASET HEALTH INDICATORS
    # ==========================================================

    # Indicates whether the dataset has a large number of features
    is_high_dimensional: bool = False

    # Indicates whether the target variable is highly imbalanced
    is_imbalanced: bool = False

    # Indicates whether the dataset contains a significant amount of missing data
    has_many_missing_values: bool = False

    # Indicates whether one or more categorical features have very high cardinality
    has_high_cardinality: bool = False

    # Indicates whether significant outliers are present
    has_outliers: bool = False

    # Indicates whether duplicate rows exist
    has_duplicates: bool = False

    # Indicates whether the dataset contains mixed feature types
    has_mixed_data_types: bool = False

    # Indicates whether constant (zero variance) columns exist
    has_constant_features: bool = False

    # Indicates whether highly correlated features exist
    has_highly_correlated_features: bool = False

    # ==========================================================
    # REPORTS
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
        Updates the last modified timestamp.
        """
        self.updated_at = datetime.now()

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

    def summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the dataset profile.
        """

        return {
            "dataset_name": self.dataset_name,
            "rows": self.num_rows,
            "columns": self.num_columns,
            "missing_percentage": self.missing_percentage,
            "duplicate_rows": self.duplicate_rows,
            "quality_score": self.quality_score,
            "target_column": self.target_column,
        }

    def __repr__(self) -> str:
        """
        Returns a readable representation of the DatasetProfile.
        """

        return (
            f"DatasetProfile("
            f"dataset='{self.dataset_name}', "
            f"rows={self.num_rows}, "
            f"columns={self.num_columns}, "
            f"missing={self.missing_percentage:.2f}%, "
            f"quality={self.quality_score})"
        )