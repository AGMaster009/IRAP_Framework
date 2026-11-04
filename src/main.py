"""
main.py
-------
Main entry point and orchestrator for the entire IRAP Framework.
Executes all 10 phases from raw dataset ingestion to final validation,
explainability audit, and report generation.
"""

import argparse
import logging
import os
import sys

# Ensure repository root is on sys.path regardless of execution mode
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader.dataset_loader import DatasetLoader
from src.profiling.dataset_profiler import DatasetProfiler
from src.profiling.feature_profiler import FeatureProfiler
from src.relationship.relationship_discovery import RelationshipDiscovery
from src.decision.decision_engine import IntelligentDecisionEngine
from src.preprocessing.preprocessor import Preprocessor
from src.validation.validator import PipelineValidator
from src.explainability.explainer import ExplainabilityEngine
from src.reports.report_generator import ReportGenerator


def setup_logging():
    """Sets up the logging configuration for the framework."""
    os.makedirs("outputs", exist_ok=True)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("outputs/pipeline.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_pipeline(dataset_path: str, target_column: str = None):
    logger = logging.getLogger("IRAP_Main")
    logger.info("=================================================================")
    logger.info("      INTELLIGENT RELATIONSHIP-AWARE PREPROCESSING (IRAP)        ")
    logger.info("=================================================================")
    logger.info(f"Target Dataset: {dataset_path}")
    logger.info(f"Designated Target Column: {target_column or 'Auto-Detect'}")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at: {dataset_path}")
        return None

    # PHASE 1: Data Ingestion
    logger.info("\n--- PHASE 1: Data Loading ---")
    loader = DatasetLoader()
    dataset = loader.load(dataset_path)
    logger.info(f"Ingested dataset shape: {dataset.get_shape()}")

    # PHASE 2: Dataset Profiling
    logger.info("\n--- PHASE 2: Dataset Profiling ---")
    dataset_profiler = DatasetProfiler()
    dataset = dataset_profiler.profile(dataset, target_column=target_column)
    logger.info(f"Dataset Profile: {dataset.dataset_profile.summary()}")

    # PHASE 3: Feature Profiling
    logger.info("\n--- PHASE 3: Feature Profiling ---")
    feature_profiler = FeatureProfiler()
    dataset = feature_profiler.profile(dataset)
    logger.info(f"Profiled {len(dataset.feature_profiles)} individual features.")

    # PHASE 4: Relationship Discovery (SHAP + XGBoost)
    logger.info("\n--- PHASE 4: Relationship Discovery & Knowledge Graph ---")
    relationship_discovery = RelationshipDiscovery()
    dataset = relationship_discovery.discover(dataset)
    if dataset.relationship_graph is not None:
        edges_count = dataset.relationship_graph.number_of_edges()
        nodes_count = dataset.relationship_graph.number_of_nodes()
        logger.info(f"Knowledge Graph: {nodes_count} Nodes, {edges_count} Discovered Edges.")

        # Export readable edge list for review
        with open("outputs/discovered_edges.txt", "w", encoding="utf-8") as f:
            f.write("IRAP Framework: Discovered SHAP Dependencies (Edges)\n")
            f.write("======================================================\n")
            edges = sorted(
                dataset.relationship_graph.edges(data=True),
                key=lambda x: x[2].get("weight", 0),
                reverse=True,
            )
            for u, v, data in edges:
                weight = data.get("weight", 0.0)
                f.write(f"[{u}] -> [{v}] (SHAP Interaction Weight: {weight:.4f})\n")

    # PHASE 5: Intelligent Decision Engine
    logger.info("\n--- PHASE 5: Intelligent Decision Engine ---")
    decision_engine = IntelligentDecisionEngine()
    dataset = decision_engine.generate_plan(dataset)
    logger.info(f"Generated dynamic preprocessing plan for {len(dataset.preprocessing_plan)} features.")

    # PHASE 6 & 7: Preprocessing Execution (Imputation, Guardrails, Encoding, Scaling)
    logger.info("\n--- PHASE 6 & 7: Context-Aware Preprocessing Execution ---")
    preprocessor = Preprocessor()
    dataset = preprocessor.process(dataset)
    logger.info(f"Processed DataFrame Shape: {dataset.processed_dataframe.shape}")

    # PHASE 8: Pipeline Validation & Ablation Study
    logger.info("\n--- PHASE 8: Pipeline Validation & Ablation Study ---")
    validator = PipelineValidator()
    dataset = validator.validate(dataset, target_column=target_column)
    v_results = dataset.validation_results

    # PHASE 9: Explainability Engine
    logger.info("\n--- PHASE 9: Explainability & Audit Engine ---")
    explainer = ExplainabilityEngine()
    dataset = explainer.explain(dataset)

    # PHASE 10: Final Academic Report Generation
    logger.info("\n--- PHASE 10: Final Report Generation ---")
    report_gen = ReportGenerator()
    report_paths = report_gen.generate(dataset)

    logger.info("=================================================================")
    logger.info("              IRAP PIPELINE EXECUTION SUMMARY                    ")
    logger.info("=================================================================")
    if v_results:
        b_m = v_results.get("baseline_metrics", {})
        i_m = v_results.get("irap_metrics", {})
        imp = v_results.get("improvements", {})
        task = v_results.get("task_type", "Classification").upper()
        logger.info(f"Task Evaluated: {task} (Target: '{v_results.get('target_column')}')")
        if task == "CLASSIFICATION":
            logger.info(f"Baseline Accuracy:  {b_m.get('accuracy', 'N/A')}")
            logger.info(f"IRAP Accuracy:      {i_m.get('accuracy', 'N/A')}  (Gain = {imp.get('delta_accuracy_percent', 0.0):+}%)")
            logger.info(f"Baseline F1-Score:  {b_m.get('f1_score', 'N/A')}")
            logger.info(f"IRAP F1-Score:      {i_m.get('f1_score', 'N/A')}  (Gain = {imp.get('delta_f1_percent', 0.0):+}%)")
        else:
            logger.info(f"Baseline RMSE:      {b_m.get('rmse', 'N/A')}")
            logger.info(f"IRAP RMSE:          {i_m.get('rmse', 'N/A')}")

        fidelity = v_results.get("imputation_fidelity", {})
        if fidelity:
            logger.info(f"Imputation Error Reduction: {fidelity.get('rmse_reduction_percent', 0.0)}% over Standard Mean Imputation")

    logger.info(f"Markdown Report: {report_paths.get('markdown')}")
    logger.info(f"HTML Report:     {report_paths.get('html')}")
    logger.info("=================================================================")
    return dataset


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="IRAP Framework CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/realistic_sample.csv",
        help="Path to the dataset file (CSV/Excel)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name for supervised tasks",
    )
    args = parser.parse_args()

    run_pipeline(dataset_path=args.dataset, target_column=args.target)


if __name__ == "__main__":
    main()
