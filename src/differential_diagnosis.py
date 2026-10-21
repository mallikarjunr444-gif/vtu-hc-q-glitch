"""
Rare Disease Categorization and Differential Diagnosis Engine.

Maps multimodal clinical, genomic, and phenotypic feature patterns
to specific rare disease classes based on clinical phenotypic ontologies
(OMIM / Orphanet aligned).

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import numpy as np
from typing import Dict, List, Any


class RareDiseaseCategorizer:
    """
    Differential diagnosis engine mapping multimodal vectors to
    specific rare-disease categories:

    1. Laminopathies & Progeroid Syndromes (e.g., Hutchinson-Gilford Progeria)
    2. Lysosomal Storage Disorders (e.g., Gaucher, Fabry, Pompe)
    3. Hereditary Connective Tissue & Skeletal Dysplasias (e.g., Ehlers-Danlos, Marfan)
    4. Inborn Errors of Metabolism & Mitochondrial Encephalopathies
    """

    CATEGORIES = {
        "laminopathy": {
            "name": "Laminopathy & Progeroid Syndrome",
            "icd10": "E34.8 / Q87.1",
            "primary_biomarkers": ["LMNA mutation", "Elevated Progerin", "Telomere Shortening"],
            "phenotype": "Premature aging, alopecia, skin sclerosis, severe growth retardation",
            "urgency": "High — Specialized pediatric genetics consultation recommended",
        },
        "lysosomal": {
            "name": "Lysosomal Storage Disorder",
            "icd10": "E75.2 (Gaucher / Fabry / Niemann-Pick)",
            "primary_biomarkers": ["Enzyme Deficiency", "Thrombocytopenia", "Hepatosplenomegaly"],
            "phenotype": "Chronic fatigue, bone crises, renal insufficiency, elevated CRP",
            "urgency": "High — Enzyme replacement therapy (ERT) evaluation needed",
        },
        "connective_tissue": {
            "name": "Connective Tissue & Skeletal Dysplasia",
            "icd10": "Q79.6 (Ehlers-Danlos) / Q87.4 (Marfan)",
            "primary_biomarkers": ["Collagen Disruption", "Joint Hypermobility", "Aortic Root Dilatation"],
            "phenotype": "Joint dislocation, tissue fragility, vascular fragility, scoliosis",
            "urgency": "Moderate-High — Cardiovascular echocardiography indicated",
        },
        "metabolic": {
            "name": "Inborn Error of Metabolism / Mitochondrial",
            "icd10": "E88.8 / G71.3",
            "primary_biomarkers": ["Lactate Elevation", "DNA Damage Index", "Oxidative Stress"],
            "phenotype": "Exercise intolerance, myopathy, metabolic acidosis, encephalopathy",
            "urgency": "Moderate — Metabolic panel & mitochondrial DNA sequencing recommended",
        },
    }

    @staticmethod
    def classify_category(features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify a patient profile into disease categories and calculate
        category affinity scores.
        """
        # Feature extraction
        lmna = features.get("lmna_mutation_score", 0.0)
        progerin = features.get("progerin_level", 0.0)
        telomere = features.get("telomere_length", 7.0)
        growth = features.get("growth_retardation_score", 0.0)
        skin = features.get("skin_elasticity", 5.0)
        joint = features.get("joint_mobility", 5.0)
        crp = features.get("c_reactive_protein", 1.0)
        platelets = features.get("platelet_count", 250.0)
        dna_damage = features.get("dna_damage_index", 0.2)
        ox_stress = features.get("oxidative_stress", 1.0)
        cv_risk = features.get("cardiovascular_risk", 2.0)

        def _norm(val, max_scale=10.0):
            val = float(val)
            if val > 1.0:
                return min(val / max_scale, 1.0)
            return max(0.0, min(val, 1.0))

        lmna_norm = _norm(lmna, 10.0)
        progerin_norm = _norm(progerin, 10.0)
        telomere_norm = max(0.0, min((10.0 - float(telomere)) / 8.0, 1.0))
        growth_norm = _norm(growth, 1.0)

        skin_abnormal = 1.0 - _norm(skin, 1.0)
        joint_abnormal = 1.0 - _norm(joint, 1.0)

        crp_norm = min(float(crp) / 30.0, 1.0)
        plt_abnormal = max(0.0, min((250.0 - float(platelets)) / 200.0, 1.0))

        dna_norm = min(float(dna_damage) / 2.0, 1.0)
        ox_norm = min(float(ox_stress) / 3.0, 1.0)
        cv_norm = _norm(cv_risk, 1.0)

        # Affinity scores (0.0 to 1.0)
        laminopathy_score = (
            0.40 * lmna_norm +
            0.35 * progerin_norm +
            0.15 * telomere_norm +
            0.10 * growth_norm
        )

        lysosomal_score = (
            0.40 * crp_norm +
            0.35 * plt_abnormal +
            0.15 * ox_norm +
            0.10 * cv_norm
        )

        connective_score = (
            0.45 * skin_abnormal +
            0.45 * joint_abnormal +
            0.10 * growth_norm
        )

        metabolic_score = (
            0.45 * dna_norm +
            0.40 * ox_norm +
            0.15 * cv_norm
        )

        scores = {
            "laminopathy": float(np.clip(laminopathy_score, 0.01, 0.99)),
            "lysosomal": float(np.clip(lysosomal_score, 0.01, 0.99)),
            "connective_tissue": float(np.clip(connective_score, 0.01, 0.99)),
            "metabolic": float(np.clip(metabolic_score, 0.01, 0.99)),
        }

        # Normalize to probability distribution
        total = sum(scores.values())
        norm_scores = {k: v / total for k, v in scores.items()}

        best_category_key = max(norm_scores, key=norm_scores.get)
        best_info = RareDiseaseCategorizer.CATEGORIES[best_category_key]

        return {
            "primary_category": best_info["name"],
            "category_key": best_category_key,
            "icd10": best_info["icd10"],
            "primary_biomarkers": best_info["primary_biomarkers"],
            "phenotypic_description": best_info["phenotype"],
            "clinical_urgency": best_info["urgency"],
            "confidence": float(norm_scores[best_category_key]),
            "affinity_breakdown": {
                RareDiseaseCategorizer.CATEGORIES[k]["name"]: float(v)
                for k, v in norm_scores.items()
            },
        }
