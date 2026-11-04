"""
preprocessing package
---------------------
Modules for context-aware missing value imputation, biological guardrails,
adaptive categorical encoding, and adaptive numerical scaling.
"""

from src.preprocessing.imputer import ContextAwareImputer
from src.preprocessing.encoders import AdaptiveEncoder
from src.preprocessing.scalers import AdaptiveScaler
from src.preprocessing.preprocessor import Preprocessor

__all__ = [
    "ContextAwareImputer",
    "AdaptiveEncoder",
    "AdaptiveScaler",
    "Preprocessor",
]

