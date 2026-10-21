"""
Classical ML baseline models for rare-disease classification.

Provides SVM, Random Forest, and XGBoost baselines
for benchmarking against quantum models.

Author: Mohammed Adil
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

from .config import CONFIG

logger = logging.getLogger(__name__)


class ClassicalBaselines:
    """
    Classical machine learning models serving as performance
    baselines for comparison with quantum classifiers.
    """

    def __init__(self, config=None):
        self.cfg = config or CONFIG.classical
        self.models = {}
        self.results = {}
        self._build_models()

    def _build_models(self):
        """Initialize all classical baseline models."""
        self.models = {
            "SVM (RBF)": SVC(
                kernel=self.cfg.svm_kernel,
                C=self.cfg.svm_C,
                probability=True,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=self.cfg.rf_max_depth,
                random_state=42,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=self.cfg.xgb_n_estimators,
                learning_rate=self.cfg.xgb_learning_rate,
                max_depth=self.cfg.xgb_max_depth,
                random_state=42
            ),
        }
        logger.info(f"Initialized {len(self.models)} classical baselines")

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all classical models on the training data."""
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            logger.info(f"  {name} train accuracy: {train_acc:.4f}")

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models and return metrics.

        Returns:
            Dictionary mapping model names to their metrics.
        """
        self.results = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            self.results[name] = metrics

            logger.info(
                f"  {name}: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_score']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}"
            )

        return self.results

    def get_predictions(
        self, model_name: str, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get predictions and probabilities from a specific model."""
        model = self.models[model_name]
        return {
            "predictions": model.predict(X),
            "probabilities": model.predict_proba(X)[:, 1],
        }

    def get_feature_importance(self, model_name: str = "Random Forest") -> Optional[np.ndarray]:
        """Extract feature importances from tree-based models."""
        model = self.models.get(model_name)
        if model and hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return None
