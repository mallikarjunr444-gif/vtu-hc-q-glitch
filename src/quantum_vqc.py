"""
Variational Quantum Classifier (VQC) for rare-disease prediction.

Implements a parameterized quantum circuit with trainable
rotation gates optimized via classical-quantum hybrid loop.

Author: Mallanagouda M
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Qiskit imports with graceful fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit.primitives import StatevectorSampler
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not installed. VQC will use classical MLP fallback.")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

from .config import CONFIG


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC) implementation.

    Architecture:
        1. Feature Map: ZZFeatureMap encodes classical data into
           quantum states via parameterized rotations and entanglement.
        2. Ansatz: RealAmplitudes variational circuit with trainable
           parameters optimized during training.
        3. Measurement: Pauli-Z expectation values decoded into
           class predictions.

    The VQC learns a decision boundary in quantum Hilbert space
    by optimizing variational parameters to minimize classification loss.
    """

    def __init__(
        self,
        n_qubits: Optional[int] = None,
        feature_map_reps: Optional[int] = None,
        ansatz_reps: Optional[int] = None,
        entanglement: Optional[str] = None,
        optimizer_name: Optional[str] = None,
        max_iter: Optional[int] = None,
    ):
        self.n_qubits = n_qubits or CONFIG.quantum.n_qubits
        self.feature_map_reps = feature_map_reps or CONFIG.quantum.feature_map_reps
        self.ansatz_reps = ansatz_reps or CONFIG.quantum.ansatz_reps
        self.entanglement = entanglement or CONFIG.quantum.entanglement
        self.optimizer_name = optimizer_name or CONFIG.quantum.optimizer_name
        self.max_iter = max_iter or CONFIG.quantum.max_iter
        self.use_quantum = HAS_QISKIT

        self.feature_map = None
        self.ansatz = None
        self.optimizer = None
        self.classifier = None
        self.training_history = []
        self.is_trained = False

        if self.use_quantum:
            self._build_vqc()
        else:
            self._build_classical_fallback()

    def _build_feature_map(self) -> QuantumCircuit:
        """Build the quantum feature map for data encoding."""
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement=self.entanglement
        )
        return self.feature_map

    def _build_ansatz(self) -> QuantumCircuit:
        """Build the variational ansatz with trainable parameters."""
        self.ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.ansatz_reps,
            entanglement=self.entanglement
        )
        return self.ansatz

    def _get_optimizer(self):
        """Select the classical optimizer for parameter updates."""
        optimizers = {
            "cobyla": COBYLA(maxiter=self.max_iter),
            "spsa": SPSA(maxiter=self.max_iter),
        }
        self.optimizer = optimizers.get(
            self.optimizer_name.lower(),
            COBYLA(maxiter=self.max_iter)
        )
        return self.optimizer

    def _build_vqc(self):
        """Assemble the full VQC pipeline."""
        logger.info(
            f"Building VQC: {self.n_qubits} qubits, "
            f"ansatz_reps={self.ansatz_reps}, "
            f"optimizer={self.optimizer_name}"
        )

        feature_map = self._build_feature_map()
        ansatz = self._build_ansatz()
        optimizer = self._get_optimizer()
        sampler = StatevectorSampler()

        # Callback to track training progress
        def callback(weights, obj_value):
            self.training_history.append(float(obj_value))
            if len(self.training_history) % 20 == 0:
                logger.info(
                    f"  VQC iteration {len(self.training_history)}: "
                    f"loss={obj_value:.4f}"
                )

        self.classifier = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler,
            callback=callback,
        )

        total_params = ansatz.num_parameters
        circuit_depth = feature_map.depth() + ansatz.depth()
        logger.info(f"VQC trainable parameters: {total_params}")
        logger.info(f"Total circuit depth: {circuit_depth}")

    def _build_classical_fallback(self):
        """Classical MLP fallback when Qiskit is unavailable."""
        logger.warning("Using classical MLP as VQC fallback")
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=500,
            random_state=42
        )

    def get_circuit_diagram(self) -> Optional[str]:
        """Return text diagram of the full VQC circuit."""
        if self.feature_map is not None and self.ansatz is not None:
            full_circuit = self.feature_map.compose(self.ansatz)
            return full_circuit.draw(output="text").__str__()
        return None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the VQC on quantum-encoded training data.

        The optimizer iteratively adjusts ansatz parameters to
        minimize the classification objective function.
        """
        logger.info(f"Training VQC on {X_train.shape[0]} samples...")
        self.training_history = []
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        logger.info(
            f"VQC training complete after {len(self.training_history)} iterations"
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate class predictions."""
        assert self.is_trained, "Model must be trained before prediction"
        return self.classifier.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        assert self.is_trained, "Model must be trained before prediction"
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_test)
        # VQC score method as fallback
        scores = self.classifier.score(X_test, np.zeros(len(X_test)))
        return np.column_stack([1 - scores, scores])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate VQC performance and return metrics."""
        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

        # ROC-AUC if probabilities are available
        try:
            y_proba = self.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics["roc_auc"] = 0.0

        label = "VQC (Quantum)" if self.use_quantum else "MLP (fallback)"
        logger.info(
            f"  {label}: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        return metrics

    def get_training_history(self):
        """Return the loss history from training."""
        return self.training_history
