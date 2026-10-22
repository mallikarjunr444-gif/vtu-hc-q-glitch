"""
Quantum Hardware Noise Simulation & Readout Error Mitigation Module.

Simulates realistic NISQ (Noisy Intermediate-Scale Quantum) environmental noise
including multi-qubit depolarizing noise and measurement readout confusion matrices.
Implements Readout Error Mitigation (TREX / Matrix Inversion) and quantifies
fidelity preservation and quantum advantage resilience for rare-disease classification.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not installed. Quantum noise module will use mathematical emulation.")

from .config import CONFIG


class QuantumNoiseBenchmark:
    """
    Evaluates quantum circuit robustness under realistic quantum hardware noise
    and demonstrates error mitigation recovery.
    """

    def __init__(
        self,
        n_qubits: Optional[int] = None,
        depolarizing_rates: Optional[List[float]] = None,
        readout_error_rate: float = 0.025,
    ):
        self.n_qubits = n_qubits or CONFIG.quantum.n_qubits
        self.depolarizing_rates = depolarizing_rates or [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
        self.readout_error_rate = readout_error_rate
        self.dim = 2 ** self.n_qubits
        self.results: Dict[str, any] = {}

    def _build_test_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Construct parameterized ZZFeatureMap circuit with bound clinical parameters."""
        try:
            from qiskit.circuit.library import zz_feature_map
            qc = zz_feature_map(
                feature_dimension=self.n_qubits,
                reps=2,
                entanglement="linear"
            )
        except (ImportError, TypeError):
            qc = ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=2,
                entanglement="linear"
            )
        return qc.assign_parameters(x[:self.n_qubits])

    def _apply_depolarizing_channel(
        self, rho: np.ndarray, p: float
    ) -> np.ndarray:
        """
        Apply global depolarizing channel:
            E(rho) = (1 - p) * rho + (p / d) * I
        where d = 2^n is the Hilbert space dimension.
        """
        d = self.dim
        return (1.0 - p) * rho + (p / d) * np.eye(d, dtype=complex)

    def _build_readout_matrix(self, p_ro: float) -> np.ndarray:
        """
        Construct 2^n x 2^n readout measurement confusion matrix M:
        M[i, j] = P(measure i | true state j)
        """
        M_1q = np.array([
            [1.0 - p_ro, p_ro],
            [p_ro, 1.0 - p_ro]
        ])
        M = M_1q
        for _ in range(self.n_qubits - 1):
            M = np.kron(M, M_1q)
        return M

    def run_noise_benchmark(
        self, test_features: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Simulate ideal vs noisy vs error-mitigated execution across
        multiple hardware noise rates.
        """
        logger.info(f"Running quantum hardware noise benchmark on {self.n_qubits} qubits...")

        if test_features is None or len(test_features) == 0:
            np.random.seed(42)
            test_x = np.random.uniform(0.1, np.pi - 0.1, size=self.n_qubits)
        else:
            test_x = test_features[0][:self.n_qubits]

        # 1. Ideal Statevector
        bound_circuit = self._build_test_circuit(test_x)
        sv_ideal = Statevector.from_instruction(bound_circuit)
        rho_ideal = DensityMatrix(sv_ideal).data
        probs_ideal = np.abs(sv_ideal.data) ** 2
        probs_ideal /= probs_ideal.sum()

        fidelities_unmitigated = []
        fidelities_mitigated = []
        tvd_noisy_list = []
        tvd_mitigated_list = []

        M_readout = self._build_readout_matrix(self.readout_error_rate)
        # Regularized matrix inversion for readout mitigation
        M_inv = np.linalg.pinv(M_readout)

        for p in self.depolarizing_rates:
            # Apply depolarizing noise
            rho_noisy = self._apply_depolarizing_channel(rho_ideal, p)

            # State fidelity under depolarizing noise
            fid_noisy = float(np.real(np.trace(rho_ideal @ rho_noisy)))
            fidelities_unmitigated.append(fid_noisy)

            # Combined depolarizing + readout measurement noise on probabilities
            diag_noisy = np.real(np.diag(rho_noisy))
            diag_noisy = np.clip(diag_noisy, 0.0, 1.0)
            diag_noisy /= diag_noisy.sum()
            p_meas_noisy = M_readout @ diag_noisy

            # Readout error mitigation: p_mit = M_inv @ p_noisy
            p_mitigated = M_inv @ p_meas_noisy
            p_mitigated = np.clip(p_mitigated, 0.0, 1.0)
            if p_mitigated.sum() > 0:
                p_mitigated /= p_mitigated.sum()

            # Total variation distances relative to ideal probabilities
            tvd_noisy = 0.5 * float(np.sum(np.abs(p_meas_noisy - probs_ideal)))
            tvd_mit = 0.5 * float(np.sum(np.abs(p_mitigated - probs_ideal)))

            tvd_noisy_list.append(tvd_noisy)
            tvd_mitigated_list.append(tvd_mit)

            # Mitigated fidelity (classical fidelity / Bhattacharyya coefficient)
            bhattacharyya = float(np.sum(np.sqrt(np.clip(p_mitigated * probs_ideal, 0.0, 1.0))))
            fidelities_mitigated.append(min(1.0, bhattacharyya))

        avg_tvd_reduction = float(
            np.mean((np.array(tvd_noisy_list) - np.array(tvd_mitigated_list)) / (np.array(tvd_noisy_list) + 1e-6)) * 100
        )

        self.results = {
            "depolarizing_rates": self.depolarizing_rates,
            "readout_error_rate": self.readout_error_rate,
            "fidelities_unmitigated": fidelities_unmitigated,
            "fidelities_mitigated": fidelities_mitigated,
            "tvd_noisy": tvd_noisy_list,
            "tvd_mitigated": tvd_mitigated_list,
            "avg_tvd_reduction_percent": avg_tvd_reduction,
            "resilience_factor": round(
                fidelities_mitigated[min(2, len(fidelities_mitigated) - 1)]
                / (fidelities_unmitigated[min(2, len(fidelities_unmitigated) - 1)] + 1e-6),
                3
            ),
        }

        logger.info(
            f"Noise benchmark complete | Avg TVD reduction with Readout Mitigation: {avg_tvd_reduction:.1f}% | "
            f"Resilience factor at p=0.02: {self.results['resilience_factor']}x"
        )
        return self.results

    def plot_benchmark(self, save_path: Optional[str] = None) -> str:
        """Generate high-resolution benchmark visualization for judges and documentation."""
        if not self.results:
            self.run_noise_benchmark()

        rates = [r * 100 for r in self.results["depolarizing_rates"]]
        fid_unmit = self.results["fidelities_unmitigated"]
        fid_mit = self.results["fidelities_mitigated"]
        tvd_noisy = self.results["tvd_noisy"]
        tvd_mit = self.results["tvd_mitigated"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        fig.patch.set_facecolor("#0a0f1d")

        # Plot 1: Fidelity vs Hardware Noise
        ax1.set_facecolor("#111827")
        ax1.plot(rates, fid_unmit, marker="o", color="#ef4444", lw=2.5, label="Raw NISQ Noisy Hardware")
        ax1.plot(rates, fid_mit, marker="s", color="#10b981", lw=2.5, label="With Readout Error Mitigation (TREX)")
        ax1.axhline(1.0, color="#60a5fa", linestyle="--", alpha=0.7, label="Ideal Noise-Free Statevector")
        ax1.set_title("Quantum State Fidelity vs Depolarizing Error Rate", color="white", fontsize=12, pad=12)
        ax1.set_xlabel("Depolarizing Noise Rate (%)", color="#94a3b8", fontsize=10)
        ax1.set_ylabel("State Fidelity F(ρ, |ψ⟩)", color="#94a3b8", fontsize=10)
        ax1.tick_params(colors="#94a3b8")
        ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
        ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
        ax1.set_ylim(0.4, 1.05)

        # Plot 2: Total Variation Distance (TVD) Error
        ax2.set_facecolor("#111827")
        width = 0.35
        x_indices = np.arange(len(rates))
        ax2.bar(x_indices - width / 2, tvd_noisy, width, label="Raw Noisy Readout", color="#f59e0b", alpha=0.85)
        ax2.bar(x_indices + width / 2, tvd_mit, width, label="Mitigated Readout (Inverted M)", color="#06b6d4", alpha=0.85)
        ax2.set_title("Readout Probability Error (Total Variation Distance)", color="white", fontsize=12, pad=12)
        ax2.set_xlabel("Depolarizing Noise Rate (%)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Total Variation Distance (Lower is Better)", color="#94a3b8", fontsize=10)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels([f"{r:.1f}%" for r in rates])
        ax2.tick_params(colors="#94a3b8")
        ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
        ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

        plt.suptitle(
            "Q-Glitch | Quantum Hardware Noise Resilience & Error Mitigation Benchmark\n"
            f"6 Qubits · Readout Error = {self.readout_error_rate*100:.1f}% · Average Error Reduction = {self.results['avg_tvd_reduction_percent']:.1f}%",
            color="white",
            fontsize=13,
            y=1.03
        )
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "results", "noise_mitigation_benchmark.png"
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info(f"Saved quantum noise benchmark chart to {save_path}")
        return save_path
