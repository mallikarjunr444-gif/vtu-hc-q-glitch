"""
Model evaluation, comparison, and reporting module.

Generates side-by-side benchmark results comparing
quantum vs classical classifiers.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, List, Optional

from .config import CONFIG

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Compares quantum and classical model performance with
    publication-quality visualizations and metric tables.
    """

    METRIC_NAMES = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def add_result(self, model_name: str, metrics: Dict[str, float]):
        """Register a model's evaluation metrics."""
        self.results[model_name] = metrics
        logger.info(f"Registered results for: {model_name}")

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Build a comparison DataFrame of all registered models.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for model_name, metrics in self.results.items():
            row = {"Model": model_name}
            for metric in self.METRIC_NAMES:
                row[metric.replace("_", " ").title()] = metrics.get(metric, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        df = df.sort_values("Accuracy", ascending=False)
        return df

    def get_best_model(self, metric: str = "f1_score") -> str:
        """Return the name of the best-performing model by given metric."""
        if not self.results:
            return "No models evaluated"
        return max(self.results, key=lambda m: self.results[m].get(metric, 0))

    def plot_comparison_bar(self, save: bool = True) -> plt.Figure:
        """
        Generate grouped bar chart comparing all models.

        Returns:
            matplotlib Figure object.
        """
        df = self.get_comparison_table()
        if df.empty:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df.index))
        width = 0.15
        colors = ["#7C3AED", "#06B6D4", "#F59E0B", "#10B981", "#EF4444"]

        for i, col in enumerate(df.columns):
            offset = (i - len(df.columns) / 2) * width
            bars = ax.bar(x + offset, df[col], width, label=col, color=colors[i % len(colors)])
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7
                )

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Q-Glitch: Quantum vs Classical Model Comparison",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=15, ha="right")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "model_comparison.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved comparison chart to {path}")

        return fig

    def plot_training_history(
        self, history: List[float], model_name: str = "VQC", save: bool = True
    ) -> plt.Figure:
        """Plot VQC training loss convergence curve."""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(history, color="#7C3AED", linewidth=2, alpha=0.8)
        ax.fill_between(range(len(history)), history, alpha=0.1, color="#7C3AED")
        ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Q-Glitch: {model_name} Training Convergence",
            fontsize=14, fontweight="bold"
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, f"{model_name.lower()}_training.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved training plot to {path}")

        return fig

    def generate_report(self) -> str:
        """Generate a text summary report of all results."""
        df = self.get_comparison_table()
        best = self.get_best_model("f1_score")

        report = [
            "=" * 60,
            "  Q-GLITCH MODEL EVALUATION REPORT",
            f"  Team: Q-Glitch | ID: {CONFIG.team_id}",
            "=" * 60,
            "",
            "PERFORMANCE COMPARISON",
            "-" * 40,
            df.to_string() if not df.empty else "No results available",
            "",
            f"Best model (by F1-score): {best}",
            "",
            "=" * 60,
        ]

        report_text = "\n".join(report)

        # Save report
        path = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(path, "w") as f:
            f.write(report_text)
        logger.info(f"Saved evaluation report to {path}")

        return report_text
