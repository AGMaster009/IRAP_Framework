"""
dataset.py
-----------

Central Dataset object for the IRAP Framework.

This class acts as the primary container that moves through every stage
of the Intelligent Relationship-Aware Preprocessing (IRAP) pipeline.

Every module in the framework receives a Dataset object, enriches it
with additional knowledge, and returns the same object.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Dataset:
    """
    Central Dataset object used throughout the IRAP Framework.

    This object stores:
    - Raw dataset
    - Processed dataset
    - Metadata
    - Profiling information
    - Relationship information
    - Knowledge graph
    - Preprocessing decisions
    - Validation results
    - Explainability information
    """

    # ==========================================================
    # RAW DATA
    # ==========================================================

    dataframe: Optional[pd.DataFrame] = None
    processed_dataframe: Optional[pd.DataFrame] = None

    # ==========================================================
    # DATASET METADATA
    # ==========================================================

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Example:
    # {
    #     "filename": "heart.csv",
    #     "file_type": "csv",
    #     "encoding": "utf-8",
    #     "separator": ",",
    #     "rows": 303,
    #     "columns": 14
    # }

    # ==========================================================
    # PROFILING
    # ==========================================================

    dataset_profile: Optional[Any] = None

    feature_profiles: Dict[str, Any] = field(default_factory=dict)

    # Example:
    # {
    #     "Age": FeatureProfile(...),
    #     "BloodPressure": FeatureProfile(...)
    # }

    # ==========================================================
    # RELATIONSHIP DISCOVERY
    # ==========================================================

    relationships: Dict[str, Any] = field(default_factory=dict)

    relationship_graph: Optional[Any] = None

    # ==========================================================
    # DECISION ENGINE
    # ==========================================================

    preprocessing_plan: List[Any] = field(default_factory=list)

    # ==========================================================
    # PREPROCESSING
    # ==========================================================

    preprocessing_history: List[Any] = field(default_factory=list)

    # ==========================================================
    # PIPELINE
    # ==========================================================

    pipeline_history: List[str] = field(default_factory=list)

    # ==========================================================
    # VALIDATION
    # ==========================================================

    validation_results: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================

    explanations: List[Any] = field(default_factory=list)

    # ==========================================================
    # DATA QUALITY
    # ==========================================================

    quality_score: Optional[float] = None

    # ==========================================================
    # TIMESTAMPS
    # ==========================================================

    created_at: datetime = field(default_factory=datetime.now)

    modified_at: datetime = field(default_factory=datetime.now)

    # ==========================================================
    # HELPER METHODS
    # ==========================================================

    def update_modified_time(self) -> None:
        """
        Updates the modified timestamp.

        Every module should call this after modifying
        the Dataset object.
        """
        self.modified_at = datetime.now()

    def has_data(self) -> bool:
        """
        Returns True if the raw dataset has been loaded.
        """
        return self.dataframe is not None

    def has_processed_data(self) -> bool:
        """
        Returns True if preprocessing has been completed.
        """
        return self.processed_dataframe is not None

    def reset_processing(self) -> None:
        """
        Clears all preprocessing-related information while
        keeping the raw dataset intact.
        """

        self.processed_dataframe = None
        self.preprocessing_plan.clear()
        self.preprocessing_history.clear()
        self.validation_results.clear()
        self.explanations.clear()
        self.quality_score = None

        self.update_modified_time()

    def get_shape(self) -> tuple:
        """
        Returns the shape of the raw dataset.

        Returns
        -------
        tuple
            (rows, columns)
        """

        if self.dataframe is None:
            return (0, 0)

        return self.dataframe.shape

    def summary(self) -> Dict[str, Any]:
        """
        Returns a lightweight summary of the Dataset object.
        Useful for logging and debugging.
        """

        rows, cols = self.get_shape()

        return {
            "rows": rows,
            "columns": cols,
            "has_profile": self.dataset_profile is not None,
            "feature_profiles": len(self.feature_profiles),
            "relationships": len(self.relationships),
            "processed": self.has_processed_data(),
            "quality_score": self.quality_score,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    def get_sampled_data(self, max_rows: int = 10000) -> pd.DataFrame:
        """
        Returns a sample of the dataset if it exceeds max_rows to prevent
        Out-Of-Memory errors during computationally heavy tasks.
        Preserves class distribution if target_column is known in dataset_profile.
        """
        if self.dataframe is None:
            return pd.DataFrame()
            
        if len(self.dataframe) <= max_rows:
            return self.dataframe.copy()
            
        target_col = None
        if self.dataset_profile and hasattr(self.dataset_profile, "target_column"):
            target_col = self.dataset_profile.target_column
            
        # Stratified sampling if target is categorical/boolean
        if target_col and target_col in self.dataframe.columns:
            target_type = str(self.dataframe[target_col].dtype)
            if target_type in ['object', 'category', 'bool'] or self.dataframe[target_col].nunique() < 20:
                try:
                    return self.dataframe.groupby(target_col, group_keys=False).apply(
                        lambda x: x.sample(frac=max_rows/len(self.dataframe), random_state=42)
                    ).copy()
                except ValueError:
                    pass # Fallback to random sample if stratification fails
                    
        return self.dataframe.sample(n=max_rows, random_state=42).copy()

    def __repr__(self) -> str:
        """
        Custom string representation of the Dataset object.
        """

        rows, cols = self.get_shape()

        return (
            f"Dataset(rows={rows}, "
            f"columns={cols}, "
            f"processed={self.has_processed_data()}, "
            f"quality_score={self.quality_score})"
        )