"""
validation package
------------------
Validation and benchmarking modules for the IRAP Framework.
Compares IRAP against standard Scikit-Learn baseline pipelines.
"""

from src.validation.validator import PipelineValidator

__all__ = ["PipelineValidator"]

