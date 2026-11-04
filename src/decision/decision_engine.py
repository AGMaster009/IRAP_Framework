import logging
from typing import Dict, Any

from src.models.dataset import Dataset
from src.models.feature_profile import FeatureProfile

class IntelligentDecisionEngine:
    """
    Phase 5: Intelligent Decision Engine
    
    This module analyzes the statistical feature profiles and the 
    SHAP-based Knowledge Graph to deterministically generate a 
    context-aware preprocessing plan for each feature.
    """

    def __init__(self, missing_drop_threshold: float = 50.0, high_cardinality_threshold: int = 15):
        self.logger = logging.getLogger("IRAP_DecisionEngine")
        self.missing_drop_threshold = missing_drop_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

    def generate_plan(self, dataset: Dataset) -> Dataset:
        self.logger.info("Starting Intelligent Decision Engine Phase...")

        if not dataset.feature_profiles:
            self.logger.warning("No feature profiles found. Run Feature Profiler first.")
            return dataset

        preprocessing_plan = []
        graph = dataset.relationship_graph

        for feature_name, profile in dataset.feature_profiles.items():
            feature_plan = {
                "feature": feature_name,
                "actions": []
            }

            # ==========================================
            # 1. Missing Value Strategy
            # ==========================================
            if profile.missing_percentage > self.missing_drop_threshold:
                action = {
                    "step": "imputation",
                    "method": "DropColumn",
                    "reason": f"Missing percentage ({profile.missing_percentage:.2f}%) exceeds threshold ({self.missing_drop_threshold}%)"
                }
                feature_plan["actions"].append(action)
                profile.recommended_imputation = "DropColumn"
            elif profile.missing_percentage > 0:
                # Check for dependencies in Knowledge Graph
                # Incoming edges mean this feature (target) is influenced by others
                has_dependencies = False
                if graph is not None and graph.has_node(feature_name):
                    in_degree = graph.in_degree(feature_name)
                    has_dependencies = in_degree > 0

                if has_dependencies:
                    # Get the predictors that influence this feature
                    predictors = [u for u, v in graph.in_edges(feature_name)]
                    action = {
                        "step": "imputation",
                        "method": "ContextAwareKNN",
                        "predictors": predictors,
                        "reason": f"Feature has missing values and {len(predictors)} discovered dependencies in Knowledge Graph."
                    }
                    profile.recommended_imputation = "ContextAwareKNN"
                else:
                    method = "MedianImputer" if profile.is_numeric() else "ModeImputer"
                    action = {
                        "step": "imputation",
                        "method": method,
                        "reason": "Feature has missing values but no discovered dependencies. Falling back to simple imputation."
                    }
                    profile.recommended_imputation = method
                feature_plan["actions"].append(action)

            # ==========================================
            # 2. Categorical Encoding Strategy
            # ==========================================
            # Only encode if we didn't decide to drop it
            if profile.recommended_imputation != "DropColumn" and profile.is_categorical():
                if profile.unique_values > self.high_cardinality_threshold:
                    method = "TargetEncoding" if hasattr(dataset.dataset_profile, 'target_column') and dataset.dataset_profile.target_column else "FrequencyEncoding"
                    reason = f"High cardinality detected ({profile.unique_values} unique values > threshold {self.high_cardinality_threshold}). Avoiding OneHotEncoding to prevent memory explosion."
                else:
                    method = "OneHotEncoding"
                    reason = f"Low cardinality ({profile.unique_values} unique values). Safe for OneHotEncoding."
                
                action = {
                    "step": "encoding",
                    "method": method,
                    "reason": reason
                }
                feature_plan["actions"].append(action)
                profile.recommended_encoding = method

            # ==========================================
            # 3. Scaling & Outlier Strategy
            # ==========================================
            if profile.recommended_imputation != "DropColumn" and profile.is_numeric():
                # If there are outliers or high skewness, use RobustScaler
                is_skewed = profile.skewness is not None and abs(profile.skewness) > 1.0
                has_outliers = profile.outlier_count > 0

                if has_outliers or is_skewed:
                    method = "RobustScaler"
                    reason = f"Feature is numeric with outliers ({profile.outlier_count}) or high skewness. RobustScaler prevents distortion."
                elif profile.distribution_type == "Gaussian":
                    method = "StandardScaler"
                    reason = "Feature is numeric and normally distributed. StandardScaler is optimal."
                else:
                    method = "MinMaxScaler"
                    reason = "Feature is numeric without heavy outliers or strict normality. Defaulting to MinMaxScaler."
                
                action = {
                    "step": "scaling",
                    "method": method,
                    "reason": reason
                }
                feature_plan["actions"].append(action)
                profile.recommended_scaling = method

            # Save the plan for this feature
            preprocessing_plan.append(feature_plan)

        # Attach plan to dataset
        dataset.preprocessing_plan = preprocessing_plan
        dataset.pipeline_history.append("Intelligent Decision Engine generated preprocessing plan.")
        dataset.update_modified_time()

        self.logger.info(f"Intelligent Decision Engine generated plan for {len(preprocessing_plan)} features.")

        return dataset
