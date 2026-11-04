"""
helpers.py
----------
Utility functions and helper algorithms for the IRAP Framework.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def ensure_directory(path: str) -> None:
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def clean_column_name(col: str) -> str:
    """Standardizes column names by stripping special characters and whitespace."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", col).strip("_").lower()


def compute_entropy(series: pd.Series) -> float:
    """Computes Shannon Entropy for a categorical series."""
    clean_s = series.dropna()
    if clean_s.empty:
        return 0.0
    probs = clean_s.value_counts(normalize=True)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def calculate_iqr_outliers(series: pd.Series, factor: float = 1.5) -> int:
    """Calculates outlier count using Tukey's Interquartile Range method."""
    clean_s = series.dropna()
    if len(clean_s) < 4:
        return 0
    q25, q75 = clean_s.quantile(0.25), clean_s.quantile(0.75)
    iqr = q75 - q25
    if iqr == 0:
        return 0
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)
    return int(((clean_s < lower_bound) | (clean_s > upper_bound)).sum())


class ExecutionTimer:
    """Context manager for timing pipeline modules."""

    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time

