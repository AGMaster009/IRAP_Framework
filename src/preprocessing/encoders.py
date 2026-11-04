"""
encoders.py
-----------
Adaptive Categorical Encoders for the IRAP Framework.

Executes intelligent categorical encoding based on cardinality and target availability:
- OneHotEncoding: Low cardinality (< 15 categories) to avoid the curse of dimensionality.
- TargetEncoding: High cardinality with a supervised target column (with smoothing).
- FrequencyEncoding: High cardinality unsupervised/frequency-based representation.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.dataset import Dataset


class AdaptiveEncoder:
    """
    Applies adaptive categorical encoding strategies to preserve data structure
    while preventing exponential memory/dimensional explosion.
    """

    def __init__(self, smoothing_weight: float = 10.0):
        self.logger = logging.getLogger("IRAP_AdaptiveEncoder")
        self.smoothing_weight = smoothing_weight
        self.encoding_maps: Dict[str, Any] = {}

    def encode(self, dataset: Dataset) -> Dataset:
        """
        Executes categorical encoding according to dataset.preprocessing_plan.
        """
        self.logger.info("Starting Adaptive Categorical Encoding...")

        if not dataset.has_data():
            self.logger.warning("No data found to encode.")
            return dataset

        df = dataset.dataframe.copy()
        plan = dataset.preprocessing_plan or []
        target_col = None
        if dataset.dataset_profile and hasattr(dataset.dataset_profile, "target_column"):
            target_col = dataset.dataset_profile.target_column

        encoded_features_count = 0

        for feature_plan in plan:
            col_name = feature_plan["feature"]
            if col_name not in df.columns:
                continue

            for action in feature_plan.get("actions", []):
                if action.get("step") != "encoding":
                    continue

                method = action.get("method")
                self.logger.info(f"Encoding '{col_name}' using {method}...")

                if method == "OneHotEncoding":
                    # Perform One-Hot Encoding
                    dummies = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True, dtype=float)
                    df.drop(columns=[col_name], inplace=True)
                    df = pd.concat([df, dummies], axis=1)
                    self.encoding_maps[col_name] = {
                        "method": "OneHotEncoding",
                        "generated_columns": list(dummies.columns)
                    }
                    encoded_features_count += 1
                    dataset.preprocessing_history.append(
                        f"One-Hot Encoded '{col_name}' into {len(dummies.columns)} binary columns."
                    )

                elif method == "TargetEncoding" and target_col and target_col in df.columns:
                    # Target Encoding with Bayesian smoothing
                    global_mean = df[target_col].mean()
                    grouped = df.groupby(col_name)[target_col]
                    counts = grouped.count()
                    means = grouped.mean()

                    # Smoothed target encoding formula: (count * mean + weight * global_mean) / (count + weight)
                    smooth_map = (counts * means + self.smoothing_weight * global_mean) / (counts + self.smoothing_weight)
                    mapping_dict = smooth_map.to_dict()

                    df[col_name] = df[col_name].map(mapping_dict).fillna(global_mean).astype(float)
                    self.encoding_maps[col_name] = {
                        "method": "TargetEncoding",
                        "mapping": mapping_dict,
                        "global_mean": float(global_mean)
                    }
                    encoded_features_count += 1
                    dataset.preprocessing_history.append(
                        f"Target Encoded '{col_name}' against '{target_col}' with smoothing={self.smoothing_weight}."
                    )

                elif method in ["FrequencyEncoding", "TargetEncoding"]:
                    # Frequency Encoding (also fallback if target_column is not available)
                    freq_map = (df[col_name].value_counts() / len(df)).to_dict()
                    df[col_name] = df[col_name].map(freq_map).fillna(0.0).astype(float)
                    self.encoding_maps[col_name] = {
                        "method": "FrequencyEncoding",
                        "mapping": freq_map
                    }
                    encoded_features_count += 1
                    dataset.preprocessing_history.append(
                        f"Frequency Encoded '{col_name}' ({len(freq_map)} unique categories)."
                    )

        dataset.dataframe = df
        dataset.pipeline_history.append(
            f"Adaptive Categorical Encoding completed: {encoded_features_count} feature(s) transformed."
        )
        dataset.update_modified_time()

        self.logger.info(f"Adaptive Encoding finished. Transformed {encoded_features_count} features.")
        return dataset

