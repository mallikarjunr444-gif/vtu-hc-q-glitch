"""
Model serialization and artifact persistence manager.

Handles saving and loading of preprocessed pipelines, classical models,
quantum kernel matrices, and VQC parameters for instant zero-latency production inference.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import os
import pickle
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def save_trained_pipeline(bundle: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """Save trained pipeline components to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = filepath or os.path.join(MODELS_DIR, "qglitch_bundle.pkl")

    # Serialize bundle
    with open(target_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Successfully saved trained model bundle to {target_path}")
    return target_path


def load_trained_pipeline(filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load pre-trained pipeline bundle from disk."""
    target_path = filepath or os.path.join(MODELS_DIR, "qglitch_bundle.pkl")

    if not os.path.exists(target_path):
        logger.warning(f"No trained model bundle found at {target_path}")
        return None

    try:
        with open(target_path, "rb") as f:
            bundle = pickle.load(f)
        logger.info(f"Loaded trained model bundle from {target_path}")
        return bundle
    except Exception as e:
        logger.error(f"Error loading model bundle: {e}")
        return None
