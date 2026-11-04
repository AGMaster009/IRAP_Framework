"""
preprocessor.py
---------------
Main Orchestrator for Phase 6 & Phase 7 of the IRAP Framework.

Executes the end-to-end preprocessing pipeline:
1. Context-Aware Missing Value Imputation (with Biological Guardrails enforcement)
2. Adaptive Categorical Encoding (Preventing dimensional explosion)
3. Adaptive Numerical Scaling (Preventing outlier distortion)
4. Stores the final clean machine-learning ready matrix in dataset.processed_dataframe
"""

import logging
from typing import Optional

import pandas as pd

from src.models.dataset import Dataset
from src.preprocessing.imputer import ContextAwareImputer
from src.preprocessing.encoders import AdaptiveEncoder
from src.preprocessing.scalers import AdaptiveScaler


class Preprocessor:
    """
    Orchestrates the execution of the entire Preprocessing Phase.
    """

    def __init__(
        self,
        guardrails_path: str = "datasets/biological_guardrails.json",
        knn_neighbors: int = 5,
        target_encoding_smoothing: float = 10.0,
    ):
        self.logger = logging.getLogger("IRAP_Preprocessor")
        self.imputer = ContextAwareImputer(guardrails_path=guardrails_path, n_neighbors=knn_neighbors)
        self.encoder = AdaptiveEncoder(smoothing_weight=target_encoding_smoothing)
        self.scaler = AdaptiveScaler()

    def process(self, dataset: Dataset) -> Dataset:
        """
        Executes the full preprocessing pipeline on the dataset according to
        the Intelligent Decision Engine's plan.
        """
        self.logger.info("==================================================")
        self.logger.info("Starting Phase 6 & 7: IRAP Preprocessing Execution")
        self.logger.info("==================================================")

        if not dataset.has_data():
            self.logger.error("Dataset has no data to preprocess.")
            return dataset

        if not dataset.preprocessing_plan:
            self.logger.warning("No preprocessing plan found. Decision Engine must run first.")
            return dataset

        # Store an untouched backup of raw dataframe if not already saved
        if "raw_dataframe" not in dataset.metadata:
            dataset.metadata["raw_dataframe"] = dataset.dataframe.copy()

        # Step 1: Imputation + Biological Guardrails
        self.logger.info("Step 1/3: Context-Aware Imputation & Guardrail Enforcement...")
        dataset = self.imputer.impute(dataset)

        # Step 2: Adaptive Categorical Encoding
        self.logger.info("Step 2/3: Adaptive Categorical Encoding...")
        dataset = self.encoder.encode(dataset)

        # Step 3: Adaptive Numerical Scaling
        self.logger.info("Step 3/3: Adaptive Numerical Scaling...")
        dataset = self.scaler.scale(dataset)

        # Final Verification & Storage into processed_dataframe
        processed_df = dataset.dataframe.copy()

        # Handle any remaining residual NaNs with standard median/mode fallback if any column was newly created
        if processed_df.isna().sum().sum() > 0:
            self.logger.warning(
                f"Residual NaNs ({processed_df.isna().sum().sum()}) detected. Applying fallback safety fill."
            )
            for c in processed_df.columns:
                if processed_df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(processed_df[c]):
                        processed_df[c] = processed_df[c].fillna(processed_df[c].median())
                    else:
                        mode_s = processed_df[c].mode()
                        processed_df[c] = processed_df[c].fillna(mode_s.iloc[0] if not mode_s.empty else 0)

        dataset.processed_dataframe = processed_df
        dataset.pipeline_history.append(
            f"Preprocessing Phase completed. Final processed shape: {dataset.processed_dataframe.shape}."
        )
        dataset.update_modified_time()

        self.logger.info(
            f"IRAP Preprocessing Complete. Processed shape: {dataset.processed_dataframe.shape}, "
            f"Residual missing values: {dataset.processed_dataframe.isna().sum().sum()}."
        )

        return dataset

