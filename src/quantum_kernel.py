"""
Quantum Support Vector Machine (QSVM) using Qiskit quantum kernels.

Implements quantum kernel computation via ZZFeatureMap and
FidelityQuantumKernel for rare-disease classification.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Qiskit imports with graceful fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.primitives import StatevectorSampler
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not installed. Quantum kernel will use classical SVM fallback.")

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

from .config import CONFIG


class QuantumKernelClassifier:
    """
    Quantum kernel-based classifier using Qiskit.

    The quantum kernel computes:
        K(x_i, x_j) = |<φ(x_i)|φ(x_j)>|²

    where φ is a quantum feature map that encodes classical data
    into quantum state space via parameterized circuits.

    This approach exploits the exponential dimensionality of Hilbert
    space to find complex decision boundaries that classical kernels
    may miss, particularly useful for small, high-dimensional
    rare-disease datasets.
    """

    def __init__(
        self,
        n_qubits: Optional[int] = None,
        feature_map_reps: Optional[int] = None,
        entanglement: Optional[str] = None,
    ):
        self.n_qubits = n_qubits or CONFIG.quantum.n_qubits
        self.feature_map_reps = feature_map_reps or CONFIG.quantum.feature_map_reps
        self.entanglement = entanglement or CONFIG.quantum.entanglement
        self.use_quantum = HAS_QISKIT
        self.feature_map = None
        self.kernel = None
        self.classifier = None
        self.is_trained = False

        if self.use_quantum:
            self._build_quantum_kernel()
        else:
            self._build_classical_fallback()

    def _build_quantum_kernel(self):
        """Construct the quantum feature map and kernel."""
        logger.info(
            f"Building quantum kernel: {self.n_qubits} qubits, "
            f"{self.feature_map_reps} reps, {self.entanglement} entanglement"
        )

        # ZZFeatureMap creates entangling feature encoding
        # It encodes data as: U_Φ(x) = exp(i Σ φ_k(x) Z_k Z_l)
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement=self.entanglement
        )

        # Build the fidelity-based quantum kernel
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )

        # SVC with precomputed quantum kernel matrix
        self.classifier = SVC(
            kernel="precomputed",
            probability=True,
            random_state=42
        )

        logger.info(f"Quantum feature map circuit depth: {self.feature_map.depth()}")
        logger.info(f"Feature map parameters: {self.feature_map.num_parameters}")

    def _build_classical_fallback(self):
        """Classical SVM fallback when Qiskit is unavailable."""
        logger.warning("Using classical RBF-SVM as fallback")
        self.classifier = SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )

    def get_circuit_diagram(self) -> Optional[str]:
        """Return text representation of the quantum feature map circuit."""
        if self.feature_map is not None:
            return self.feature_map.draw(output="text").__str__()
        return None

    def compute_kernel_matrix(
        self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the quantum kernel matrix.

        For training: K_train[i,j] = |<φ(x_i)|φ(x_j)>|²
        For testing:  K_test[i,j]  = |<φ(x_test_i)|φ(x_train_j)>|²
        """
        if not self.use_quantum:
            raise RuntimeError("Quantum kernel not available without Qiskit")

        if X_test is None:
            logger.info(f"Computing training kernel matrix ({X_train.shape[0]}x{X_train.shape[0]})")
            return self.kernel.evaluate(X_train)
        else:
            logger.info(
                f"Computing test kernel matrix "
                f"({X_test.shape[0]}x{X_train.shape[0]})"
            )
            return self.kernel.evaluate(X_test, X_train)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the quantum kernel classifier.

        For quantum mode: computes kernel matrix, then fits SVC.
        For classical fallback: directly fits SVC with RBF kernel.
        """
        logger.info(f"Training QSVM on {X_train.shape[0]} samples...")

        if self.use_quantum:
            self._X_train = X_train  # Store for test kernel computation
            K_train = self.compute_kernel_matrix(X_train)
            self.classifier.fit(K_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

        self.is_trained = True
        logger.info("QSVM training complete")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test data."""
        assert self.is_trained, "Model must be trained before prediction"

        if self.use_quantum:
            K_test = self.compute_kernel_matrix(self._X_train, X_test)
            return self.classifier.predict(K_test)
        return self.classifier.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Generate probability predictions for test data."""
        assert self.is_trained, "Model must be trained before prediction"

        if self.use_quantum:
            K_test = self.compute_kernel_matrix(self._X_train, X_test)
            return self.classifier.predict_proba(K_test)
        return self.classifier.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quantum kernel classifier.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        label = "Quantum Kernel QSVC" if self.use_quantum else "Classical SVM (fallback)"
        logger.info(
            f"  {label}: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        return metrics
