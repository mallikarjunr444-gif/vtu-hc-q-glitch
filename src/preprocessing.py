"""
Data preprocessing and feature engineering pipeline.

Handles normalization, imputation, dimensionality reduction,
and feature selection for quantum circuit compatibility.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from .config import CONFIG

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessing pipeline that prepares multimodal clinical data
    for both classical ML and quantum circuit encoding.

    Quantum circuits have limited qubits, so we must reduce
    feature dimensions while preserving discriminative information.
    """

    def __init__(self, n_quantum_features: Optional[int] = None):
        """
        Args:
            n_quantum_features: Target dimensionality for quantum encoding.
                Defaults to quantum config n_qubits.
        """
        self.n_quantum_features = n_quantum_features or CONFIG.quantum.n_qubits
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler(feature_range=(0, np.pi))
        self.imputer = SimpleImputer(strategy="median")
        self.pca = PCA(n_components=self.n_quantum_features)
        self.selector = SelectKBest(f_classif, k=self.n_quantum_features)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataPreprocessor":
        """
        Fit all preprocessing transformers on training data.

        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Training labels
        """
        logger.info(f"Fitting preprocessor on {X.shape} data")

        # Step 1: Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Step 2: Standardize features (zero mean, unit variance)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Step 3: Feature selection (statistical)
        self.selector.fit(X_scaled, y)

        # Step 4: PCA for quantum-compatible dimensionality
        X_selected = self.selector.transform(X_scaled)
        self.pca.fit(X_selected)

        self._fitted = True
        variance_explained = self.pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA variance explained: {variance_explained:.3f} "
            f"with {self.n_quantum_features} components"
        )
        return self

    def transform_classical(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data for classical ML models.
        Returns standardized features without dimension reduction.
        """
        assert self._fitted, "Call fit() before transform"
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

    def transform_quantum(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data for quantum circuit encoding.

        Returns features reduced to n_qubits dimensions and
        normalized to [0, pi] range for rotation gate encoding.
        """
        assert self._fitted, "Call fit() before transform"
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_selected = self.selector.transform(X_scaled)
        X_pca = self.pca.transform(X_selected)

        # Normalize to [0, pi] for quantum rotation gates
        X_quantum = self.normalizer.fit_transform(X_pca)
        return X_quantum

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and return both classical and quantum-ready features.

        Returns:
            (X_classical, X_quantum) tuple
        """
        self.fit(X, y)
        X_classical = self.transform_classical(X)
        X_quantum = self.transform_quantum(X)
        return X_classical, X_quantum

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature selection scores for interpretability."""
        assert self._fitted, "Call fit() before getting importance"
        scores = self.selector.scores_
        pvalues = self.selector.pvalues_

        return pd.DataFrame({
            "feature_index": range(len(scores)),
            "f_score": scores,
            "p_value": pvalues,
            "selected": [
                i in self.selector.get_support(indices=True)
                for i in range(len(scores))
            ]
        }).sort_values("f_score", ascending=False)
