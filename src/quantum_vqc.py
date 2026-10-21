"""
Variational Quantum Classifier (VQC) for rare-disease prediction.

Implements a parameterized quantum circuit (ZZFeatureMap + RealAmplitudes)
with trainable rotation gates optimized via classical-quantum hybrid loop (COBYLA).

Author: Mallanagouda M
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Qiskit imports with native statevector and fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from scipy.optimize import minimize
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit or Scipy not installed. VQC will use classical MLP fallback.")

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
        1. Feature Map: ZZFeatureMap encodes classical clinical/genomic features
           into quantum state space via non-linear entanglement.
        2. Ansatz: RealAmplitudes parameterized circuit with trainable rotation
           angles (Ry gates and CX entangling gates).
        3. Measurement: Pauli-Z expectation value on qubit 0 decoded into
           rare-disease risk probability: p = (1 + <Z_0>) / 2.
        4. Optimization: COBYLA classical optimizer minimizing binary cross-entropy.
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
        self.max_iter = max_iter or min(CONFIG.quantum.max_iter, 50)
        self.use_quantum = HAS_QISKIT

        self.feature_map = None
        self.ansatz = None
        self.full_circuit = None
        self.obs_op = None
        self.weights = None
        self.training_history: List[float] = []
        self.is_trained = False
        self.fallback_model = None

        if self.use_quantum:
            self._build_vqc()
        else:
            self._build_classical_fallback()

    def _build_vqc(self):
        """Assemble the parameterized quantum circuit and measurement observable."""
        logger.info(
            f"Building VQC: {self.n_qubits} qubits, "
            f"ansatz_reps={self.ansatz_reps}, "
            f"optimizer={self.optimizer_name}"
        )

        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.feature_map_reps,
            entanglement=self.entanglement
        )

        self.ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.ansatz_reps,
            entanglement=self.entanglement
        )

        self.full_circuit = self.feature_map.compose(self.ansatz)
        # Pauli-Z observable on qubit 0: 'ZII...I'
        self.obs_op = SparsePauliOp(["Z" + "I" * (self.n_qubits - 1)])

        total_params = self.ansatz.num_parameters
        circuit_depth = self.full_circuit.depth()
        logger.info(f"VQC trainable parameters: {total_params}")
        logger.info(f"Total circuit depth: {circuit_depth}")

    def _build_classical_fallback(self):
        """Classical MLP fallback when Qiskit is unavailable."""
        logger.warning("Using classical MLP as VQC fallback")
        self.fallback_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=500,
            random_state=42
        )

    def get_circuit_diagram(self) -> Optional[str]:
        """Return text diagram of the full VQC circuit."""
        if self.full_circuit is not None:
            return self.full_circuit.draw(output="text").__str__()
        return None

    def _predict_prob_single(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Compute predicted probability for a single sample x given variational weights theta."""
        param_dict = dict(zip(self.feature_map.parameters, x))
        param_dict.update(dict(zip(self.ansatz.parameters, theta)))
        bound = self.full_circuit.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound)
        exp_val = float(sv.expectation_value(self.obs_op).real)
        # Map expectation in [-1, 1] to probability in [0, 1]
        prob = 0.5 * (1.0 + exp_val)
        return float(np.clip(prob, 1e-6, 1.0 - 1e-6))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the VQC on quantum-encoded training data using classical optimization.
        """
        logger.info(f"Training VQC on {X_train.shape[0]} samples...")
        self.training_history = []

        if not self.use_quantum:
            self.fallback_model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("VQC (classical MLP fallback) training complete")
            return

        # Subsample for fast variational training if cohort is large
        train_size = min(len(X_train), 50)
        indices = np.random.RandomState(42).choice(len(X_train), train_size, replace=False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]

        n_params = self.ansatz.num_parameters
        init_weights = np.random.RandomState(42).normal(0, 0.1, size=n_params)

        def cost_function(theta):
            probs = np.array([self._predict_prob_single(x, theta) for x in X_sub])
            # Binary Cross Entropy loss
            bce = -np.mean(y_sub * np.log(probs) + (1 - y_sub) * np.log(1 - probs))
            self.training_history.append(float(bce))
            return float(bce)

        res = minimize(
            cost_function,
            init_weights,
            method="COBYLA",
            options={"maxiter": self.max_iter}
        )

        self.weights = res.x
        self.is_trained = True
        logger.info(
            f"VQC training complete after {len(self.training_history)} evaluations | final loss: {res.fun:.4f}"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Scikit-learn style fit method."""
        return self.train(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Generate class probability predictions."""
        assert self.is_trained, "Model must be trained before prediction"

        if not self.use_quantum:
            return self.fallback_model.predict_proba(X_test)

        probs_pos = np.array([self._predict_prob_single(x, self.weights) for x in X_test])
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate binary class predictions (threshold 0.5)."""
        proba = self.predict_proba(X_test)[:, 1]
        return (proba >= 0.5).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate VQC performance and return metrics."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        label = "VQC (Quantum RealAmplitudes)" if self.use_quantum else "MLP (fallback)"
        logger.info(
            f"  {label}: Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}"
        )
        return metrics

    def get_training_history(self) -> List[float]:
        """Return the loss history from training."""
        return self.training_history
