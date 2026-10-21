"""
Unit tests for Q-Glitch quantum and classical pipeline.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np

from src.config import CONFIG, ProjectConfig
from src.data_generator import RareDiseaseDataGenerator, prepare_splits
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalBaselines
from src.quantum_kernel import QuantumKernelClassifier
from src.quantum_vqc import VariationalQuantumClassifier
from src.evaluation import ModelEvaluator


class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_default_config(self):
        cfg = ProjectConfig()
        self.assertEqual(cfg.project_name, "Q-Glitch")
        self.assertEqual(cfg.team_id, "VTU-QFF-HC-0190")
        self.assertEqual(cfg.quantum.n_qubits, 6)

    def test_quantum_config_defaults(self):
        self.assertEqual(CONFIG.quantum.feature_map_reps, 2)
        self.assertEqual(CONFIG.quantum.entanglement, "full")


class TestDataGenerator(unittest.TestCase):
    """Test synthetic data generation."""

    def test_generate_dataset(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        self.assertIn("clinical", data)
        self.assertIn("genetic", data)
        self.assertIn("phenotypic", data)
        self.assertIn("combined", data)
        self.assertIn("labels", data)
        self.assertEqual(len(data["labels"]), CONFIG.data.n_samples)

    def test_label_distribution(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        prevalence = data["labels"].mean()
        self.assertTrue(0.05 < prevalence < 0.35)

    def test_feature_dimensions(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        expected_total = (
            CONFIG.data.n_clinical_features
            + CONFIG.data.n_genetic_markers
            + CONFIG.data.n_phenotypic_features
        )
        self.assertEqual(data["combined"].shape[1], expected_total)

    def test_data_splits(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        splits = prepare_splits(data)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        total = sum(len(s[0]) for s in splits.values())
        self.assertEqual(total, CONFIG.data.n_samples)


class TestPreprocessor(unittest.TestCase):
    """Test preprocessing pipeline."""

    def setUp(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        self.splits = prepare_splits(data)

    def test_fit_transform(self):
        X_train, y_train = self.splits["train"]
        preprocessor = DataPreprocessor()
        X_cl, X_q = preprocessor.fit_transform(X_train, y_train)
        self.assertEqual(X_cl.shape[0], X_train.shape[0])
        self.assertEqual(X_q.shape[1], CONFIG.quantum.n_qubits)

    def test_quantum_range(self):
        X_train, y_train = self.splits["train"]
        preprocessor = DataPreprocessor()
        _, X_q = preprocessor.fit_transform(X_train, y_train)
        self.assertGreaterEqual(X_q.min(), -1e-5)
        self.assertLessEqual(X_q.max(), np.pi + 0.05)


class TestClassicalModels(unittest.TestCase):
    """Test classical ML baselines."""

    def setUp(self):
        gen = RareDiseaseDataGenerator()
        data = gen.generate()
        splits = prepare_splits(data)
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        preprocessor = DataPreprocessor()
        X_train_cl, _ = preprocessor.fit_transform(X_train, y_train)
        X_test_cl = preprocessor.transform_classical(X_test)

        self.classical = ClassicalBaselines()
        self.classical.train_all(X_train_cl, y_train)
        self.X_test_cl = X_test_cl
        self.y_test = y_test

    def test_models_created(self):
        self.assertEqual(len(self.classical.models), 3)

    def test_evaluation(self):
        results = self.classical.evaluate_all(self.X_test_cl, self.y_test)
        for name, metrics in results.items():
            self.assertIn("accuracy", metrics)
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertIn("f1_score", metrics)


class TestQuantumModels(unittest.TestCase):
    """Test quantum model initialization and fallbacks."""

    def test_qsvm_init(self):
        qsvm = QuantumKernelClassifier()
        self.assertEqual(qsvm.n_qubits, CONFIG.quantum.n_qubits)

    def test_vqc_init(self):
        vqc = VariationalQuantumClassifier()
        self.assertEqual(vqc.n_qubits, CONFIG.quantum.n_qubits)
        self.assertEqual(vqc.ansatz_reps, CONFIG.quantum.ansatz_reps)


class TestEvaluator(unittest.TestCase):
    """Test evaluation and reporting."""

    def test_comparison_table(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("Model A", {"accuracy": 0.9, "f1_score": 0.85})
        evaluator.add_result("Model B", {"accuracy": 0.8, "f1_score": 0.75})
        df = evaluator.get_comparison_table()
        self.assertEqual(len(df), 2)

    def test_best_model(self):
        evaluator = ModelEvaluator()
        evaluator.add_result("Model A", {"f1_score": 0.85})
        evaluator.add_result("Model B", {"f1_score": 0.95})
        self.assertEqual(evaluator.get_best_model("f1_score"), "Model B")


if __name__ == "__main__":
    if pytest is not None and "pytest" in sys.modules:
        pytest.main([__file__, "-v"])
    else:
        unittest.main(verbosity=2)
