"""
Synthetic multimodal rare-disease dataset generator.

Generates realistic clinical, genetic, and phenotypic features
for rare-disease prediction benchmarking.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

from .config import CONFIG

logger = logging.getLogger(__name__)


class RareDiseaseDataGenerator:
    """
    Generates synthetic multimodal patient data simulating
    rare-disease clinical scenarios.

    Modalities:
        - Clinical: vital signs, lab values, BMI, blood markers
        - Genetic: mutation indicators, gene expression levels
        - Phenotypic: observable trait scores and risk indicators
    """

    # Clinical feature names
    CLINICAL_FEATURES = [
        "age", "bmi", "heart_rate", "blood_pressure_sys",
        "blood_pressure_dia", "cholesterol", "hemoglobin",
        "white_blood_cells", "platelet_count", "creatinine",
        "albumin", "c_reactive_protein"
    ]

    # Genetic marker names
    GENETIC_MARKERS = [
        "lmna_mutation_score", "zmpste24_expression",
        "progerin_level", "telomere_length",
        "dna_damage_index", "autophagy_marker",
        "inflammatory_cytokine", "oxidative_stress"
    ]

    # Phenotypic feature names
    PHENOTYPIC_FEATURES = [
        "growth_retardation_score", "skin_elasticity",
        "hair_density_index", "joint_mobility",
        "cardiovascular_risk", "skeletal_abnormality"
    ]

    def __init__(self, config=None):
        self.cfg = config or CONFIG.data
        np.random.seed(self.cfg.random_seed)
        logger.info(
            f"Initializing data generator: {self.cfg.n_samples} samples, "
            f"prevalence={self.cfg.rare_disease_prevalence}"
        )

    def _generate_clinical_features(self, n: int, labels: np.ndarray) -> np.ndarray:
        """Generate clinical measurement features with disease-correlated shifts."""
        features = np.random.randn(n, self.cfg.n_clinical_features)

        # Affected patients show shifted distributions
        affected_mask = labels == 1
        n_affected = affected_mask.sum()

        # Age tends lower for progeria-like conditions
        features[affected_mask, 0] -= 1.5  # age shift
        # BMI tends lower
        features[affected_mask, 1] -= 1.2
        # Elevated inflammatory markers
        features[affected_mask, 11] += 2.0  # c_reactive_protein
        # Lower hemoglobin
        features[affected_mask, 6] -= 0.8
        # Elevated creatinine
        features[affected_mask, 9] += 0.6

        # Add realistic noise
        features += np.random.normal(0, 0.3, features.shape)
        return features

    def _generate_genetic_markers(self, n: int, labels: np.ndarray) -> np.ndarray:
        """Generate genetic marker values with mutation-linked patterns."""
        markers = np.random.randn(n, self.cfg.n_genetic_markers)

        affected_mask = labels == 1

        # LMNA mutation score elevated in affected
        markers[affected_mask, 0] += 2.5
        # Progerin accumulation
        markers[affected_mask, 2] += 2.0
        # Shorter telomere length
        markers[affected_mask, 3] -= 1.8
        # Higher DNA damage
        markers[affected_mask, 4] += 1.5
        # Inflammatory cytokine elevation
        markers[affected_mask, 6] += 1.2

        markers += np.random.normal(0, 0.4, markers.shape)
        return markers

    def _generate_phenotypic_features(self, n: int, labels: np.ndarray) -> np.ndarray:
        """Generate phenotypic observation scores."""
        features = np.random.randn(n, self.cfg.n_phenotypic_features)

        affected_mask = labels == 1

        # Growth retardation present
        features[affected_mask, 0] += 2.2
        # Reduced skin elasticity
        features[affected_mask, 1] -= 1.5
        # Lower hair density
        features[affected_mask, 2] -= 1.3
        # Reduced joint mobility
        features[affected_mask, 3] -= 1.0
        # Higher cardiovascular risk
        features[affected_mask, 4] += 2.0
        # Skeletal abnormalities
        features[affected_mask, 5] += 1.8

        features += np.random.normal(0, 0.35, features.shape)
        return features

    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete multimodal dataset.

        Returns:
            Dictionary with keys 'clinical', 'genetic', 'phenotypic',
            'combined', and 'labels'.
        """
        n = self.cfg.n_samples
        prevalence = self.cfg.rare_disease_prevalence

        # Generate labels with specified prevalence
        n_positive = int(n * prevalence)
        labels = np.zeros(n, dtype=int)
        positive_indices = np.random.choice(n, n_positive, replace=False)
        labels[positive_indices] = 1

        logger.info(f"Generated labels: {n_positive} positive, {n - n_positive} negative")

        # Generate each modality
        clinical = self._generate_clinical_features(n, labels)
        genetic = self._generate_genetic_markers(n, labels)
        phenotypic = self._generate_phenotypic_features(n, labels)

        # Create DataFrames
        df_clinical = pd.DataFrame(clinical, columns=self.CLINICAL_FEATURES)
        df_genetic = pd.DataFrame(genetic, columns=self.GENETIC_MARKERS)
        df_phenotypic = pd.DataFrame(phenotypic, columns=self.PHENOTYPIC_FEATURES)

        # Combined multimodal DataFrame
        df_combined = pd.concat([df_clinical, df_genetic, df_phenotypic], axis=1)
        df_labels = pd.Series(labels, name="rare_disease_label")

        logger.info(
            f"Dataset shape: {df_combined.shape} "
            f"(clinical={df_clinical.shape[1]}, "
            f"genetic={df_genetic.shape[1]}, "
            f"phenotypic={df_phenotypic.shape[1]})"
        )

        return {
            "clinical": df_clinical,
            "genetic": df_genetic,
            "phenotypic": df_phenotypic,
            "combined": df_combined,
            "labels": df_labels,
        }

    def save_dataset(self, data: Dict[str, pd.DataFrame], output_dir: Optional[str] = None):
        """Save generated dataset to CSV files."""
        out_dir = output_dir or self.cfg.data_dir
        os.makedirs(out_dir, exist_ok=True)

        for name, df in data.items():
            filepath = os.path.join(out_dir, f"{name}.csv")
            if isinstance(df, pd.Series):
                df.to_csv(filepath, index=False)
            else:
                df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath}")

    def load_dataset(self, data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load previously saved dataset from CSV files."""
        load_dir = data_dir or self.cfg.data_dir
        data = {}
        for name in ["clinical", "genetic", "phenotypic", "combined"]:
            filepath = os.path.join(load_dir, f"{name}.csv")
            if os.path.exists(filepath):
                data[name] = pd.read_csv(filepath)

        labels_path = os.path.join(load_dir, "labels.csv")
        if os.path.exists(labels_path):
            data["labels"] = pd.read_csv(labels_path).squeeze()

        return data


def prepare_splits(
    data: Dict[str, pd.DataFrame],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split multimodal data into train/val/test sets.

    Returns:
        Dictionary with 'train', 'val', 'test' keys,
        each containing (X, y) tuple.
    """
    X = data["combined"].values
    y = data["labels"].values

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val,
        random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Data splits — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
