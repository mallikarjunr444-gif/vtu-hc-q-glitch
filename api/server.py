"""
Q-Glitch Clinical REST API Server.

Production FastAPI backend serving hybrid quantum-classical rare-disease predictions,
multimodal categorization, and batch screening endpoints for hospital and research integration.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.differential_diagnosis import RareDiseaseCategorizer
from src.model_store import load_trained_pipeline
from src.data_generator import RareDiseaseDataGenerator

app = FastAPI(
    title="Q-Glitch Clinical API",
    description="Multimodal Hybrid Quantum-Classical AI for Rare-Disease Diagnostic Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_BUNDLE = None


def get_bundle():
    """Retrieve or initialize the model bundle."""
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        MODEL_BUNDLE = load_trained_pipeline()
    return MODEL_BUNDLE


class PatientProfile(BaseModel):
    """Patient multimodal input data schema."""
    patient_id: Optional[str] = Field("PT-1001", description="Patient identifier")
    # Clinical markers
    age: float = Field(25.0, ge=1.0, le=110.0, description="Patient age in years")
    bmi: float = Field(22.0, ge=10.0, le=60.0, description="Body Mass Index")
    heart_rate: float = Field(75.0, ge=30.0, le=220.0, description="Resting heart rate (bpm)")
    blood_pressure_sys: float = Field(120.0, ge=60.0, le=240.0, description="Systolic blood pressure")
    blood_pressure_dia: float = Field(80.0, ge=30.0, le=140.0, description="Diastolic blood pressure")
    cholesterol: float = Field(190.0, ge=80.0, le=450.0, description="Serum cholesterol (mg/dL)")
    hemoglobin: float = Field(14.0, ge=4.0, le=22.0, description="Hemoglobin (g/dL)")
    white_blood_cells: float = Field(7.0, ge=1.0, le=35.0, description="WBC count (x10^3/uL)")
    platelet_count: float = Field(250.0, ge=20.0, le=800.0, description="Platelet count (x10^3/uL)")
    creatinine: float = Field(1.0, ge=0.2, le=12.0, description="Serum creatinine (mg/dL)")
    albumin: float = Field(4.2, ge=1.0, le=6.5, description="Serum albumin (g/dL)")
    c_reactive_protein: float = Field(2.0, ge=0.0, le=100.0, description="C-Reactive Protein (mg/L)")
    # Genomic markers
    lmna_mutation_score: float = Field(1.5, ge=0.0, le=10.0, description="LMNA gene mutation score")
    zmpste24_expression: float = Field(1.0, ge=0.0, le=5.0, description="ZMPSTE24 relative expression")
    progerin_level: float = Field(0.5, ge=0.0, le=10.0, description="Serum progerin concentration")
    telomere_length: float = Field(7.2, ge=1.0, le=12.0, description="Relative telomere length (kb)")
    dna_damage_index: float = Field(0.3, ge=0.0, le=5.0, description="Gamma-H2AX DNA damage foci index")
    autophagy_marker: float = Field(1.0, ge=0.0, le=5.0, description="LC3-II / Beclin-1 expression")
    inflammatory_cytokine: float = Field(0.5, ge=0.0, le=10.0, description="IL-6 / TNF-alpha multiplex score")
    oxidative_stress: float = Field(1.0, ge=0.0, le=5.0, description="Reactive Oxygen Species marker")
    # Phenotypic features
    growth_retardation_score: float = Field(0.2, ge=0.0, le=1.0, description="Growth retardation index")
    skin_elasticity: float = Field(0.8, ge=0.0, le=1.0, description="Cutaneous elasticity coefficient")
    joint_mobility: float = Field(0.8, ge=0.0, le=1.0, description="Beighton joint mobility score")
    hair_density_index: float = Field(0.8, ge=0.0, le=1.0, description="Alopecia / hair density score")
    cardiovascular_risk: float = Field(0.3, ge=0.0, le=1.0, description="Atherosclerotic / vascular risk index")
    skeletal_abnormality: float = Field(0.2, ge=0.0, le=1.0, description="Skeletal dysplasia / bone deformity score")


@app.get("/api/meta")
def root_meta():
    """Root metadata endpoint."""
    return {
        "system": "Q-Glitch Clinical AI Engine",
        "team": "Q-Glitch",
        "team_id": "VTU-QFF-HC-0190",
        "event": "VTU-Qiskit Fall Fest Hackathon 2026",
        "status": "operational",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    """Health check endpoint confirming Qiskit and model availability."""
    bundle = get_bundle()
    return {
        "status": "healthy",
        "models_loaded": bundle is not None,
        "n_qubits": CONFIG.quantum.n_qubits,
        "quantum_feature_map": "ZZFeatureMap",
        "qiskit_version": "2.x",
    }


@app.get("/api/v1/categories")
def get_categories():
    """Return all supported rare-disease clinical categories."""
    return {
        "categories": RareDiseaseCategorizer.CATEGORIES
    }


@app.post("/api/v1/predict")
def predict_patient(patient: PatientProfile):
    """
    Run full multimodal hybrid quantum-classical diagnostic assessment
    for an individual patient profile.
    """
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not yet loaded. Run training first.")

    preprocessor = bundle["preprocessor"]
    classical_models = bundle["classical"]
    qsvm = bundle["qsvm"]
    vqc = bundle["vqc"]

    # Construct feature array (26 features matching order)
    raw_vector = np.array([[
        patient.age, patient.bmi, patient.heart_rate, patient.blood_pressure_sys,
        patient.blood_pressure_dia, patient.cholesterol, patient.hemoglobin,
        patient.white_blood_cells, patient.platelet_count, patient.creatinine,
        patient.albumin, patient.c_reactive_protein,
        patient.lmna_mutation_score, patient.zmpste24_expression,
        patient.progerin_level, patient.telomere_length,
        patient.dna_damage_index, patient.autophagy_marker,
        patient.inflammatory_cytokine, patient.oxidative_stress,
        patient.growth_retardation_score, patient.skin_elasticity,
        patient.joint_mobility, patient.hair_density_index,
        patient.cardiovascular_risk, patient.skeletal_abnormality
    ]])

    # Transform
    X_cl = preprocessor.transform_classical(raw_vector)
    X_q = preprocessor.transform_quantum(raw_vector)

    # 1. Classical predictions
    rf_pred = classical_models.get_predictions("Random Forest", X_cl)
    rf_risk = float(rf_pred["probabilities"][0])

    svm_pred = classical_models.get_predictions("SVM (RBF)", X_cl)
    svm_risk = float(svm_pred["probabilities"][0])

    gb_pred = classical_models.get_predictions("Gradient Boosting", X_cl)
    gb_risk = float(gb_pred["probabilities"][0])

    # 2. Quantum Kernel prediction (QSVC)
    qsvm_risk = float(qsvm.predict_proba(X_q)[:, 1][0])

    # 3. Variational Quantum Classifier (VQC)
    vqc_proba = float(vqc.predict_proba(X_q)[:, 1][0])

    # Ensemble consensus risk
    ensemble_risk = float(0.4 * rf_risk + 0.3 * qsvm_risk + 0.3 * vqc_proba)

    # Differential diagnosis
    features_dict = patient.model_dump()
    diff_diag = RareDiseaseCategorizer.classify_category(features_dict)

    # Triage tier
    if ensemble_risk >= 0.60:
        triage_status = "CRITICAL / HIGH RISK"
        recommendation = "Immediate referral to medical genetics and specialized rare disease center."
    elif ensemble_risk >= 0.30:
        triage_status = "MODERATE RISK"
        recommendation = "Recommend comprehensive biomarker panel and secondary screening."
    else:
        triage_status = "LOW RISK / CONTROL"
        recommendation = "Standard annual checkup. No significant rare disease indicators detected."

    return {
        "patient_id": patient.patient_id,
        "ensemble_risk_probability": round(ensemble_risk, 4),
        "triage_status": triage_status,
        "clinical_recommendation": recommendation,
        "differential_diagnosis": diff_diag,
        "model_breakdown": {
            "random_forest_risk": round(rf_risk, 4),
            "svm_rbf_risk": round(svm_risk, 4),
            "gradient_boosting_risk": round(gb_risk, 4),
            "quantum_kernel_qsvc_risk": round(qsvm_risk, 4),
            "variational_quantum_vqc_risk": round(vqc_proba, 4),
        },
        "quantum_metrics": {
            "n_qubits": CONFIG.quantum.n_qubits,
            "feature_map": "ZZFeatureMap",
            "quantum_state_rotation_angles": [round(float(a), 4) for a in X_q[0]],
        }
    }


# Mount modern interactive web platform static files at root
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
