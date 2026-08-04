"""
dataset_profiler.py
-------------------

The DatasetProfiler module is responsible for analyzing the entire dataset
to generate high-level dataset characteristics and metadata.
"""

import logging
from typing import Optional

import pandas as pd

from src.models.dataset import Dataset
from src.models.dataset_profile import DatasetProfile


class DatasetProfiler:
    """
    Analyzes a Dataset object to populate its DatasetProfile.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def profile(self, dataset: Dataset, target_column: Optional[str] = None) -> Dataset:
        """
        Profiles the dataset and populates the dataset.dataset_profile field.
        """
        
        self.logger.info("Starting dataset profiling...")

        if not dataset.has_data():
            raise ValueError("Dataset has no data loaded. Run DatasetLoader first.")

        df = dataset.dataframe
        profile = DatasetProfile()

        # Copy basic metadata from loader if available
        profile.dataset_name = dataset.metadata.get("filename", "Unknown Dataset")
        profile.file_name = dataset.metadata.get("filename", "")
        profile.file_type = dataset.metadata.get("extension", "")
        profile.file_size_mb = dataset.metadata.get("memory_usage_mb", 0.0)

        # Dimensions
        profile.num_rows, profile.num_columns = dataset.get_shape()
        profile.memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        # Missing Values
        missing_counts = df.isnull().sum()
        profile.total_missing_values = int(missing_counts.sum())
        profile.missing_percentage = (profile.total_missing_values / (profile.num_rows * profile.num_columns)) * 100 if profile.num_rows > 0 else 0
        profile.missing_by_column = missing_counts.to_dict()
        
        # Duplicates
        profile.duplicate_rows = int(df.duplicated().sum())
        profile.duplicate_percentage = (profile.duplicate_rows / profile.num_rows) * 100 if profile.num_rows > 0 else 0

        # Unique Values
        profile.unique_values = df.nunique().to_dict()

        # Data types and column groupings
        profile.data_types = df.dtypes.astype(str).to_dict()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        bool_cols = df.select_dtypes(include=['bool']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        profile.numeric_columns = len(numeric_cols)
        profile.boolean_columns = len(bool_cols)
        profile.datetime_columns = len(datetime_cols)
        profile.categorical_columns = len(categorical_cols)

        # Health Indicators
        profile.is_high_dimensional = profile.num_columns > 100
        profile.has_many_missing_values = profile.missing_percentage > 10.0
        profile.has_duplicates = profile.duplicate_rows > 0
        
        constant_cols = [col for col, count in profile.unique_values.items() if count <= 1]
        profile.has_constant_features = len(constant_cols) > 0
        
        profile.has_mixed_data_types = False # Advanced detection handled separately later
        
        # Target Information
        if target_column:
            if target_column not in df.columns:
                self.logger.warning(f"Target column '{target_column}' not found in dataset.")
            else:
                profile.target_column = target_column
                profile.target_type = str(df[target_column].dtype)
                
                # Class distribution for categorical targets
                if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category' or df[target_column].nunique() < 20:
                    profile.class_distribution = df[target_column].value_counts(normalize=True).to_dict()
                    
                    # Imbalance check
                    if len(profile.class_distribution) > 0:
                        min_class_ratio = min(profile.class_distribution.values())
                        profile.is_imbalanced = min_class_ratio < 0.1 # Example threshold: 10%

        # Attach profile
        dataset.dataset_profile = profile
        dataset.pipeline_history.append("Dataset profiled successfully.")
        
        self.logger.info("Dataset profiling completed.")
        return dataset
