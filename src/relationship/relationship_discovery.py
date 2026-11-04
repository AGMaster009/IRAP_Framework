import logging
import traceback
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any

from src.models.dataset import Dataset

# Lazy imports for heavy libraries to speed up initial load
try:
    import xgboost as xgb
    import shap
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class RelationshipDiscovery:
    """
    Phase 4: Relationship Discovery Module
    
    This module uses surrogate modeling (XGBoost) and SHAP (SHapley Additive exPlanations)
    to discover non-linear, complex relationships between features.
    
    It constructs a Context Knowledge Graph representing feature dependencies.
    """

    def __init__(self, threshold: float = 0.1, max_sample_rows: int = 5000):
        self.logger = logging.getLogger("IRAP_RelationshipDiscovery")
        self.threshold = threshold
        self.max_sample_rows = max_sample_rows

    def _prepare_data_for_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for XGBoost by label-encoding categorical columns
        while preserving NaNs so XGBoost can handle them natively.
        """
        encoded_df = df.copy()
        for col in encoded_df.columns:
            dtype_name = str(encoded_df[col].dtype)
            if dtype_name in ['object', 'category', 'string'] or encoded_df[col].apply(type).eq(str).any():
                # Factorize converts to int, NaNs become -1
                codes, _ = pd.factorize(encoded_df[col])
                # Convert back to float to allow NaNs
                codes_float = codes.astype(float)
                codes_float[codes == -1] = np.nan
                encoded_df[col] = codes_float
            elif dtype_name == 'bool':
                encoded_df[col] = encoded_df[col].astype(float)
                
        return encoded_df

    def discover(self, dataset: Dataset) -> Dataset:
        """
        Executes the Relationship Discovery phase on the dataset.
        """
        self.logger.info("Starting Relationship Discovery Phase...")

        if not dataset.has_data():
            self.logger.warning("No data found in Dataset. Skipping.")
            return dataset

        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost or SHAP is not installed. Please install dependencies.")
            return dataset

        # 1. Get a safe sample to prevent Out-Of-Memory (OOM)
        df_sample = dataset.get_sampled_data(max_rows=self.max_sample_rows)
        self.logger.info(f"Using sampled data shape {df_sample.shape} for discovery.")

        # 2. Prepare data (encode categoricals, keep NaNs)
        df_encoded = self._prepare_data_for_xgboost(df_sample)
        
        # We will build a directed graph where edges flow from Predictor -> Target
        graph = nx.DiGraph()
        relationships_dict = {}

        # 3. Iterate through every feature to act as a target
        features = df_encoded.columns.tolist()
        for feature in features:
            graph.add_node(feature)
            
        for target_col in features:
            self.logger.debug(f"Analyzing dependencies for target: {target_col}")
            
            # If target column is entirely NaN or has < 2 unique non-null values, skip
            if df_encoded[target_col].nunique(dropna=True) < 2:
                self.logger.debug(f"Skipping {target_col} due to insufficient unique values.")
                continue

            # Drop rows where the target is NaN (only for training this specific surrogate)
            valid_idx = df_encoded[target_col].notna()
            X = df_encoded.loc[valid_idx].drop(columns=[target_col])
            y = df_encoded.loc[valid_idx, target_col]
            
            if len(X) < 3: # Too few rows to train
                continue
                
            # Determine problem type (Classification vs Regression)
            # If it's boolean or small categorical, use classifier, else regressor
            is_classification = df_encoded[target_col].nunique(dropna=True) <= 20
            
            try:
                if is_classification:
                    # Some categories might be skipped due to dropping NaNs, ensure y is 0-indexed
                    y_codes, _ = pd.factorize(y)
                    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss')
                    model.fit(X, y_codes)
                else:
                    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
                    model.fit(X, y)
                    
                # Extract SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # shap_values might be a list for multi-class classification
                if isinstance(shap_values, list):
                    shap_values = np.abs(np.array(shap_values)).mean(axis=0) # average across classes
                elif len(np.array(shap_values).shape) == 3:
                    # shape is (n_samples, n_features, n_classes)
                    shap_values = np.abs(np.array(shap_values)).mean(axis=2)
                
                # Calculate mean absolute SHAP for each predictor
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Normalize SHAP scores between 0 and 1 for this target
                max_shap = mean_abs_shap.max()
                if max_shap > 0:
                    normalized_shap = mean_abs_shap / max_shap
                else:
                    normalized_shap = mean_abs_shap
                    
                # Save relationships above threshold
                target_deps = {}
                for predictor_col, score, norm_score in zip(X.columns, mean_abs_shap, normalized_shap):
                    if norm_score >= self.threshold:
                        # Add edge: Predictor -> Target (Predictor influences Target)
                        graph.add_edge(predictor_col, target_col, weight=float(norm_score), absolute_shap=float(score))
                        target_deps[predictor_col] = {
                            "normalized_score": float(norm_score),
                            "absolute_score": float(score)
                        }
                
                if target_deps:
                    relationships_dict[target_col] = target_deps
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze relationships for {target_col}: {e}\n{traceback.format_exc()}")

        # 4. Save results to Dataset object
        dataset.relationship_graph = graph
        dataset.relationships = relationships_dict
        
        dataset.pipeline_history.append(f"Relationship Discovery (SHAP + XGBoost) completed. Found {graph.number_of_edges()} edges.")
        dataset.update_modified_time()
        
        self.logger.info(f"Relationship Discovery Complete. Knowledge Graph contains {graph.number_of_edges()} dependent edges.")

        return dataset
