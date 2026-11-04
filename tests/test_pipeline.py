"""
test_pipeline.py
----------------
End-to-end integration and unit tests for the IRAP Framework.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.loader.dataset_loader import DatasetLoader
from src.profiling.dataset_profiler import DatasetProfiler
from src.profiling.feature_profiler import FeatureProfiler
from src.relationship.relationship_discovery import RelationshipDiscovery
from src.decision.decision_engine import IntelligentDecisionEngine
from src.preprocessing.preprocessor import Preprocessor
from src.validation.validator import PipelineValidator
from src.explainability.explainer import ExplainabilityEngine
from src.reports.report_generator import ReportGenerator


class TestIRAPPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv_path = "tests/test_cohort.csv"
        os.makedirs("tests", exist_ok=True)
        # Create a controlled synthetic dataset
        np.random.seed(42)
        n = 60
        ages = np.random.randint(25, 75, size=n)
        bmis = 20 + (ages * 0.1) + np.random.normal(0, 1, size=n)
        bps = 90 + (ages * 0.4) + (bmis * 1.1) + np.random.normal(0, 2, size=n)
        cats = np.random.choice(["A", "B", "C"], size=n)
        targets = ((ages > 50) & (bmis > 25)).astype(int)

        df = pd.DataFrame({
            "age": ages,
            "bmi": bmis,
            "bp": bps,
            "category": cats,
            "target": targets
        })
        # Inject missing values
        df.loc[0:5, "bmi"] = np.nan
        df.loc[10:14, "bp"] = np.nan
        df.to_csv(cls.test_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_path):
            os.remove(cls.test_csv_path)

    def test_full_pipeline_execution(self):
        # 1. Loader
        loader = DatasetLoader()
        dataset = loader.load(self.test_csv_path)
        self.assertTrue(dataset.has_data())
        self.assertEqual(dataset.get_shape()[0], 60)

        # 2. Dataset Profiling
        dataset = DatasetProfiler().profile(dataset, target_column="target")
        self.assertIsNotNone(dataset.dataset_profile)

        # 3. Feature Profiling
        dataset = FeatureProfiler().profile(dataset)
        self.assertEqual(len(dataset.feature_profiles), 5)

        # 4. Relationship Discovery
        dataset = RelationshipDiscovery().discover(dataset)
        self.assertIsNotNone(dataset.relationship_graph)
        self.assertGreater(dataset.relationship_graph.number_of_edges(), 0)

        # 5. Decision Engine
        dataset = IntelligentDecisionEngine().generate_plan(dataset)
        self.assertGreater(len(dataset.preprocessing_plan), 0)

        # 6. Preprocessing Execution
        dataset = Preprocessor().process(dataset)
        self.assertTrue(dataset.has_processed_data())
        self.assertFalse(dataset.processed_dataframe.isna().any().any())

        # 7. Validation & Ablation
        dataset = PipelineValidator().validate(dataset, target_column="target")
        self.assertIn("baseline_metrics", dataset.validation_results)
        self.assertIn("irap_metrics", dataset.validation_results)

        # 8. Explainability
        dataset = ExplainabilityEngine().explain(dataset)
        self.assertGreater(len(dataset.explanations), 0)

        # 9. Reports
        paths = ReportGenerator().generate(dataset)
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["html"]))


if __name__ == "__main__":
    unittest.main()

