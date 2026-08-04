"""
main.py
-------

Main entry point for the IRAP Framework.
"""

import logging
import os
import sys

from src.loader.dataset_loader import DatasetLoader
from src.profiling.dataset_profiler import DatasetProfiler

def setup_logging():
    """Sets up the logging configuration for the framework."""
    os.makedirs("outputs", exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("outputs/pipeline.log", mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("IRAP_Main")
    
    logger.info("Starting IRAP Framework Pipeline...")
    
    dataset_path = "datasets/sample.csv"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Sample dataset not found at {dataset_path}. Please create it to run the test.")
        return
        
    # Phase 1: Data Loading
    logger.info("--- PHASE 1: Data Loading ---")
    loader = DatasetLoader()
    try:
        dataset = loader.load(dataset_path)
        logger.info(f"Loaded dataset shape: {dataset.get_shape()}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
        
    # Phase 2: Dataset Profiling
    logger.info("--- PHASE 2: Dataset Profiling ---")
    profiler = DatasetProfiler()
    try:
        # Assuming we have a column named 'target' for the test, or None
        dataset = profiler.profile(dataset, target_column="target")
        logger.info(f"Dataset Profile Summary: {dataset.dataset_profile.summary()}")
    except Exception as e:
        logger.error(f"Failed to profile dataset: {e}")
        return
        
    logger.info("Pipeline Execution Complete.")
    
    # Print Pipeline History
    logger.info("Pipeline History:")
    for step in dataset.pipeline_history:
        logger.info(f" -> {step}")

if __name__ == "__main__":
    main()
