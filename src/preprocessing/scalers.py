"""
scalers.py
----------
Adaptive Numerical Scalers for the IRAP Framework.

Applies mathematically optimal scalers based on statistical distributions:
- RobustScaler: For skewed or outlier-heavy numeric features (median & IQR).
- StandardScaler: For Gaussian / normally distributed numeric features (mean & std).
- MinMaxScaler: For bounded or uniform numeric features without extreme outliers.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from src.models.dataset import Dataset


class AdaptiveScaler:
    """
    Applies adaptive numerical scaling to numeric features based on
    their empirical skewness, outlier counts, and distribution normality.
    """

    def __init__(self):
        self.logger = logging.getLogger("IRAP_AdaptiveScaler")
        self.fitted_scalers: Dict[str, Any] = {}

    def scale(self, dataset: Dataset) -> Dataset:
        """
        Executes numerical scaling according to dataset.preprocessing_plan.
        """
        self.logger.info("Starting Adaptive Numerical Scaling...")

        if not dataset.has_data():
            self.logger.warning("No data found to scale.")
            return dataset

        df = dataset.dataframe.copy()
        plan = dataset.preprocessing_plan or []
        target_col = None
        if dataset.dataset_profile and hasattr(dataset.dataset_profile, "target_column"):
            target_col = dataset.dataset_profile.target_column

        scaled_features_count = 0

        for feature_plan in plan:
            col_name = feature_plan["feature"]
            # Skip if column was dropped or is the target column
            if (
                col_name not in df.columns
                or col_name == target_col
                or (target_col and col_name.strip() == target_col.strip())
            ):
                continue

            for action in feature_plan.get("actions", []):
                if action.get("step") != "scaling":
                    continue

                method = action.get("method")
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    continue

                self.logger.info(f"Scaling numeric feature '{col_name}' using {method}...")

                col_values = df[[col_name]].values.astype(float)

                if method == "RobustScaler":
                    scaler = RobustScaler()
                    scaled_vals = scaler.fit_transform(col_values)
                    self.fitted_scalers[col_name] = scaler
                    df[col_name] = scaled_vals.flatten()
                    scaled_features_count += 1
                    dataset.preprocessing_history.append(
                        f"RobustScaler applied to '{col_name}' (Protected against outliers/skewness)."
                    )

                elif method == "StandardScaler":
                    scaler = StandardScaler()
                    scaled_vals = scaler.fit_transform(col_values)
                    self.fitted_scalers[col_name] = scaler
                    df[col_name] = scaled_vals.flatten()
                    scaled_features_count += 1
                    dataset.preprocessing_history.append(
                        f"StandardScaler applied to '{col_name}' (Gaussian distribution preserved)."
                    )

                elif method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    scaled_vals = scaler.fit_transform(col_values)
                    self.fitted_scalers[col_name] = scaler
                    df[col_name] = scaled_vals.flatten()
                    scaled_features_count += 1
                    dataset.preprocessing_history.append(
                        f"MinMaxScaler applied to '{col_name}' (Mapped to [0, 1] range)."
                    )

        dataset.dataframe = df
        dataset.pipeline_history.append(
            f"Adaptive Numerical Scaling completed: {scaled_features_count} feature(s) scaled."
        )
        dataset.update_modified_time()

        self.logger.info(f"Adaptive Scaling finished. Scaled {scaled_features_count} features.")
        return dataset
