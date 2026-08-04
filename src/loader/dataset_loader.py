"""
dataset_loader.py
-----------------

Loads datasets into the IRAP Framework.

Supported formats
-----------------
- CSV
- Excel (.xlsx)
- Excel (.xls)

This module creates the Dataset object that flows through
the entire IRAP Framework.
"""

import csv
import os
from pathlib import Path
from typing import Optional

import chardet
import pandas as pd

from src.models.dataset import Dataset


class DatasetLoader:
    """
    Loads datasets into the IRAP Framework.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    # ==========================================================
    # PUBLIC METHOD
    # ==========================================================

    def load(self, file_path: str) -> Dataset:
        """
        Loads a dataset and returns a Dataset object.
        """

        self.validate_file(file_path)

        extension = self.detect_file_type(file_path)

        if extension == ".csv":

            encoding = self.detect_encoding(file_path)

            separator = self.detect_separator(
                file_path,
                encoding
            )

            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=separator
            )

        else:

            dataframe = pd.read_excel(file_path)

            encoding = "N/A"

            separator = "N/A"

        dataset = Dataset()

        dataset.dataframe = dataframe

        dataset.metadata = {
            "filename": Path(file_path).name,
            "filepath": os.path.abspath(file_path),
            "extension": extension,
            "encoding": encoding,
            "separator": separator,
            "rows": dataframe.shape[0],
            "columns": dataframe.shape[1],
            "memory_usage_mb": round(
                dataframe.memory_usage(deep=True).sum()
                / (1024 * 1024),
                2
            )
        }

        dataset.pipeline_history.append("Dataset loaded successfully.")

        return dataset

    # ==========================================================
    # FILE VALIDATION
    # ==========================================================

    def validate_file(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset not found:\n{file_path}"
            )

        extension = Path(file_path).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        if os.path.getsize(file_path) == 0:
            raise ValueError("Dataset file is empty.")

    # ==========================================================
    # FILE TYPE
    # ==========================================================

    def detect_file_type(self, file_path: str):

        return Path(file_path).suffix.lower()

    # ==========================================================
    # ENCODING
    # ==========================================================

    def detect_encoding(
        self,
        file_path: str
    ) -> str:

        with open(file_path, "rb") as f:

            raw = f.read(50000)

        result = chardet.detect(raw)

        return result["encoding"] if result["encoding"] is not None else "utf-8"

    # ==========================================================
    # DELIMITER
    # ==========================================================

    def detect_separator(
        self,
        file_path: str,
        encoding: str
    ) -> str:

        with open(
            file_path,
            "r",
            encoding=encoding
        ) as f:

            sample = f.read(2048)

        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ","

    # ==========================================================
    # DATASET PREVIEW
    # ==========================================================

    def preview(
        self,
        dataset: Dataset,
        rows: int = 5
    ):

        return dataset.dataframe.head(rows)

    # ==========================================================
    # METADATA
    # ==========================================================

    def get_metadata(
        self,
        dataset: Dataset
    ):

        return dataset.metadata