"""
feature_profiler.py
-------------------

The FeatureProfiler module is responsible for analyzing every single column 
(feature) in the dataset independently to extract detailed statistical profiles.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import normaltest

from src.models.dataset import Dataset
from src.models.feature_profile import FeatureProfile

class FeatureProfiler:
    """
    Analyzes each feature in a Dataset object and populates dataset.feature_profiles.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculates Shannon Entropy for a categorical/discrete series."""
        probs = series.value_counts(normalize=True)
        if len(probs) <= 1:
            return 0.0
        return float(-np.sum(probs * np.log2(probs + 1e-9)))

    def profile(self, dataset: Dataset) -> Dataset:
        """
        Profiles every feature in the dataset.
        """
        self.logger.info("Starting feature profiling...")

        if not dataset.has_data():
            raise ValueError("Dataset has no data loaded. Run DatasetLoader first.")

        # Use the sampled data to prevent OOM on very large datasets
        # Feature profiling is mostly O(N), but max_rows=100000 keeps it fast.
        df = dataset.get_sampled_data(max_rows=100000)
        
        feature_profiles: Dict[str, FeatureProfile] = {}

        for i, column_name in enumerate(df.columns):
            series = df[column_name]
            profile = FeatureProfile(feature_name=column_name, display_name=column_name, column_index=i)
            
            # Basic stats
            profile.data_type = str(series.dtype)
            profile.missing_count = int(series.isnull().sum())
            profile.missing_percentage = (profile.missing_count / len(series)) * 100 if len(series) > 0 else 0.0
            profile.has_missing_values = profile.missing_count > 0
            
            non_null_series = series.dropna()
            profile.unique_values = int(non_null_series.nunique())
            profile.unique_percentage = (profile.unique_values / len(non_null_series)) * 100 if len(non_null_series) > 0 else 0.0
            profile.constant_feature = profile.unique_values <= 1
            
            # Numeric Analysis
            if profile.is_numeric() and not profile.is_boolean():
                try:
                    profile.mean = float(non_null_series.mean())
                    profile.median = float(non_null_series.median())
                    profile.minimum = float(non_null_series.min())
                    profile.maximum = float(non_null_series.max())
                    profile.variance = float(non_null_series.var())
                    profile.standard_deviation = float(non_null_series.std())
                    profile.range_value = profile.maximum - profile.minimum
                    
                    q1 = non_null_series.quantile(0.25)
                    q3 = non_null_series.quantile(0.75)
                    profile.interquartile_range = float(q3 - q1)
                    
                    profile.skewness = float(non_null_series.skew())
                    profile.kurtosis = float(non_null_series.kurtosis())
                    
                    # Normality Test (D'Agostino's K-squared test)
                    if len(non_null_series) >= 8:
                        stat, p_value = normaltest(non_null_series)
                        profile.normality_score = float(p_value)
                        if p_value > 0.05:
                            profile.distribution_type = "Normal"
                        else:
                            profile.distribution_type = "Non-Normal"
                    else:
                        profile.distribution_type = "Insufficient Data"
                    
                    # Basic outlier detection (IQR method)
                    lower_bound = q1 - 1.5 * profile.interquartile_range
                    upper_bound = q3 + 1.5 * profile.interquartile_range
                    outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
                    profile.outlier_count = len(outliers)
                    profile.outlier_percentage = (profile.outlier_count / len(non_null_series)) * 100 if len(non_null_series) > 0 else 0.0
                    profile.has_outliers = profile.outlier_count > 0
                    if profile.has_outliers:
                        profile.outlier_detection_method = "IQR"
                except Exception as e:
                    self.logger.warning(f"Error calculating numeric stats for {column_name}: {e}")

            # Categorical Analysis (includes text and booleans)
            elif profile.is_categorical() or profile.is_boolean():
                if len(non_null_series) > 0:
                    profile.mode = non_null_series.mode().iloc[0]
                    profile.entropy = self._calculate_entropy(non_null_series)
                    
                    # Cardinality check
                    if profile.unique_values > 100 or profile.unique_percentage > 50:
                        profile.cardinality = "High"
                    else:
                        profile.cardinality = "Low"

            feature_profiles[column_name] = profile

        dataset.feature_profiles = feature_profiles
        dataset.pipeline_history.append(f"Feature profiling completed for {len(feature_profiles)} features.")
        self.logger.info(f"Feature profiling completed for {len(feature_profiles)} features.")
        
        return dataset
