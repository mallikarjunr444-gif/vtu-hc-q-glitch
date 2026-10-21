"""
Quantum Support Vector Machine (QSVM) using Qiskit quantum kernels.

Implements quantum kernel computation via ZZFeatureMap and quantum state
fidelity overlap for multimodal rare-disease classification.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Qiskit imports with native statevector fidelity and fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not installed. Quantum kernel will use classical SVM fallback.")

# Optional Qiskit ML if available
try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT_ML = True
except ImportError:
    HAS_QISKIT_ML = False

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

    where φ is a quantum feature map (ZZFeatureMap) that encodes classical data
    into quantum Hilbert state space via parameterized entanglement gates.

    This approach exploits exponential Hilbert space to find non-linear decision
    boundaries that classical kernels struggle with on small, high-dimensional cohorts.
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
        self.classifier = None
        self.is_trained = False
        self._X_train = None
        self._train_statevectors = None

        if self.use_quantum:
            self._build_quantum_kernel()
        else:
            self._build_classical_fallback()

    def _build_quantum_kernel(self):
        """Construct the quantum feature map."""
        logger.info(
            f"Building quantum kernel: {self.n_qubits} qubits, "
            f"{self.feature_map_reps} reps, {self.entanglement} entanglement"
        )

        # ZZFeatureMap parameterized circuit: U_Φ(x) = exp(i Σ φ_k(x) Z_k Z_l)
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement=self.entanglement
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

    def _compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        """
        Map each data vector x into its corresponding quantum statevector
        |φ(x)> using the parameterized ZZFeatureMap.
        """
        n_samples = X.shape[0]
        dim = 2 ** self.n_qubits
        vectors = np.empty((n_samples, dim), dtype=np.complex128)

        for i in range(n_samples):
            bound_circuit = self.feature_map.assign_parameters(X[i])
            sv = Statevector.from_instruction(bound_circuit)
            vectors[i] = sv.data

        return vectors

    def compute_kernel_matrix(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the quantum fidelity kernel matrix:
            K[i, j] = |<φ(x1_i)|φ(x2_j)>|²
        """
        if not self.use_quantum:
            raise RuntimeError("Quantum kernel not available without Qiskit")

        if X2 is None:
            # Training kernel matrix: K(X1, X1)
            logger.info(f"Computing quantum training kernel matrix ({X1.shape[0]}x{X1.shape[0]})...")
            V1 = self._compute_statevectors(X1)
            self._train_statevectors = V1
            # Inner products: <ψ_i | ψ_j> = V1[i]*.data @ V1[j].data
            gram = np.dot(V1, V1.conj().T)
            K = np.abs(gram) ** 2
            return K
        else:
            # Test kernel matrix: K(X1, X2) -> rows correspond to X1 (test), cols to X2 (train)
            logger.info(f"Computing quantum test kernel matrix ({X1.shape[0]}x{X2.shape[0]})...")
            V1 = self._compute_statevectors(X1)
            V2 = self._train_statevectors if (self._train_statevectors is not None and len(self._train_statevectors) == len(X2)) else self._compute_statevectors(X2)
            gram = np.dot(V1, V2.conj().T)
            K = np.abs(gram) ** 2
            return K

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the quantum kernel classifier."""
        logger.info(f"Training QSVM on {X_train.shape[0]} samples...")

        if self.use_quantum:
            self._X_train = X_train
            K_train = self.compute_kernel_matrix(X_train)
            self.classifier.fit(K_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)

        self.is_trained = True
        logger.info("QSVM training complete")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Scikit-learn style fit method."""
        return self.train(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions for test data."""
        assert self.is_trained, "Model must be trained before prediction"

        if self.use_quantum:
            K_test = self.compute_kernel_matrix(X_test, self._X_train)
            return self.classifier.predict(K_test)
        return self.classifier.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Generate probability predictions for test data."""
        assert self.is_trained, "Model must be trained before prediction"

        if self.use_quantum:
            K_test = self.compute_kernel_matrix(X_test, self._X_train)
            return self.classifier.predict_proba(K_test)
        return self.classifier.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the quantum kernel classifier."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        label = "Quantum Kernel QSVC" if self.use_quantum else "Classical SVM (fallback)"
        logger.info(
            f"  {label}: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        return metrics
