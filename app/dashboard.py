"""
Q-Glitch Interactive Clinical Dashboard.

Streamlit-based web application for rare-disease risk assessment
with quantum vs classical model comparison.

Author: Mallikarjun.R
Team: Q-Glitch (VTU-QFF-HC-0190)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG
from src.data_generator import RareDiseaseDataGenerator, prepare_splits
from src.preprocessing import DataPreprocessor
from src.classical_models import ClassicalBaselines
from src.quantum_kernel import QuantumKernelClassifier
from src.quantum_vqc import VariationalQuantumClassifier
from src.evaluation import ModelEvaluator


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=CONFIG.dashboard.page_title,
    page_icon=CONFIG.dashboard.page_icon,
    layout=CONFIG.dashboard.layout,
)


def apply_custom_css():
    """Apply Q-Glitch branded styling."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .main { font-family: 'Inter', sans-serif; }

        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #24243e 100%);
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #7C3AED, #06B6D4, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }

        .hero-subtitle {
            text-align: center;
            color: #a0aec0;
            font-size: 1.1rem;
            margin-top: 0.25rem;
        }

        .metric-card {
            background: rgba(124, 58, 237, 0.1);
            border: 1px solid rgba(124, 58, 237, 0.3);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #7C3AED;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .quantum-badge {
            background: linear-gradient(90deg, #7C3AED, #06B6D4);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .disclaimer {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #EF4444;
            padding: 0.8rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.85rem;
            color: #fca5a5;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the dashboard header."""
    st.markdown('<h1 class="hero-title">⚛️ Q-Glitch</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Multimodal Quantum AI for Rare-Disease Prediction &nbsp;|&nbsp; '
        'Team Q-Glitch &nbsp;|&nbsp; VTU-QFF-HC-0190'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle" style="font-size:0.9rem;">'
        'Mallikarjun.R &bull; Mohammed Adil &bull; Mallanagouda M'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


@st.cache_data
def load_and_train_models():
    """Load data, train all models, and cache results."""
    # Generate dataset
    generator = RareDiseaseDataGenerator()
    dataset = generator.generate()
    splits = prepare_splits(dataset)

    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    # Preprocess
    preprocessor = DataPreprocessor()
    X_train_cl, X_train_q = preprocessor.fit_transform(X_train, y_train)
    X_test_cl = preprocessor.transform_classical(X_test)
    X_test_q = preprocessor.transform_quantum(X_test)

    # Classical baselines
    classical = ClassicalBaselines()
    classical.train_all(X_train_cl, y_train)
    classical_results = classical.evaluate_all(X_test_cl, y_test)

    # Quantum Kernel
    qsvm = QuantumKernelClassifier()
    qsvm.train(X_train_q, y_train)
    qsvm_metrics = qsvm.evaluate(X_test_q, y_test)

    # VQC
    vqc = VariationalQuantumClassifier()
    vqc.train(X_train_q, y_train)
    vqc_metrics = vqc.evaluate(X_test_q, y_test)

    # Collect all results
    all_results = {**classical_results}
    all_results["Quantum Kernel (QSVM)"] = qsvm_metrics
    all_results["VQC (Variational)"] = vqc_metrics

    return {
        "dataset": dataset,
        "results": all_results,
        "qsvm": qsvm,
        "vqc": vqc,
        "classical": classical,
        "preprocessor": preprocessor,
        "vqc_history": vqc.get_training_history(),
        "X_test_q": X_test_q,
        "X_test_cl": X_test_cl,
        "y_test": y_test,
    }


def render_model_comparison(results):
    """Render the model comparison section."""
    st.header("📊 Model Comparison: Quantum vs Classical")

    # Build comparison DataFrame
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics.get("accuracy", 0),
            "F1 Score": metrics.get("f1_score", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "ROC-AUC": metrics.get("roc_auc", 0),
        })
    df = pd.DataFrame(rows).set_index("Model")

    # Highlight quantum models
    st.dataframe(
        df.style.format("{:.4f}").highlight_max(axis=0, color="#7C3AED40"),
        use_container_width=True,
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df.index))
    width = 0.15
    colors = ["#7C3AED", "#06B6D4", "#F59E0B", "#10B981", "#EF4444"]

    for i, col in enumerate(df.columns):
        offset = (i - len(df.columns) / 2) * width
        bars = ax.bar(x + offset, df[col], width, label=col, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Q-Glitch: Quantum vs Classical Benchmark", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_facecolor("#1a1040")
    fig.patch.set_facecolor("#0f0c29")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#333")
    for text in ax.legend_.get_texts():
        text.set_color("white")
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)


def render_patient_assessment(cached):
    """Render the patient risk assessment input form."""
    st.header("🩺 Patient Risk Assessment")

    st.markdown(
        '<div class="disclaimer">'
        '⚠️ Research prototype only — not for clinical diagnosis.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Clinical Data")
        age = st.slider("Age", 1, 100, 25)
        bmi = st.slider("BMI", 10.0, 45.0, 22.0, step=0.5)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 160, 75)
        bp_sys = st.slider("BP Systolic", 80, 200, 120)
        bp_dia = st.slider("BP Diastolic", 40, 130, 80)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 200)

    with col2:
        st.subheader("Lab Values")
        hemoglobin = st.slider("Hemoglobin (g/dL)", 5.0, 20.0, 13.5, step=0.5)
        wbc = st.slider("White Blood Cells (×10³)", 2.0, 20.0, 7.0, step=0.5)
        platelets = st.slider("Platelet Count (×10³)", 50, 450, 250)
        creatinine = st.slider("Creatinine (mg/dL)", 0.3, 5.0, 1.0, step=0.1)
        albumin = st.slider("Albumin (g/dL)", 1.5, 5.5, 4.0, step=0.1)
        crp = st.slider("C-Reactive Protein (mg/L)", 0.0, 50.0, 3.0, step=0.5)

    with col3:
        st.subheader("Genetic & Phenotypic")
        lmna = st.slider("LMNA Mutation Score", 0.0, 10.0, 2.0, step=0.5)
        progerin = st.slider("Progerin Level", 0.0, 10.0, 1.5, step=0.5)
        telomere = st.slider("Telomere Length", 0.0, 10.0, 6.0, step=0.5)
        growth = st.slider("Growth Retardation Score", 0.0, 10.0, 1.0, step=0.5)
        skin = st.slider("Skin Elasticity", 0.0, 10.0, 7.0, step=0.5)
        cardio_risk = st.slider("Cardiovascular Risk", 0.0, 10.0, 2.0, step=0.5)

    if st.button("🔬 Run Quantum Risk Assessment", type="primary", use_container_width=True):
        # Build feature vector (26 features to match training data)
        raw_features = np.array([[
            age, bmi, heart_rate, bp_sys, bp_dia, cholesterol,
            hemoglobin, wbc, platelets, creatinine, albumin, crp,
            lmna, 5.0, progerin, telomere, 3.0, 4.0, 5.0, 3.0,
            growth, skin, 5.0, 5.0, cardio_risk, 3.0
        ]])

        preprocessor = cached["preprocessor"]
        X_q = preprocessor.transform_quantum(raw_features)
        X_cl = preprocessor.transform_classical(raw_features)

        st.markdown("---")
        st.subheader("🧬 Assessment Results")

        res_cols = st.columns(3)

        # Classical prediction
        cl_pred = cached["classical"].get_predictions("Random Forest", X_cl)
        cl_risk = cl_pred["probabilities"][0] * 100

        with res_cols[0]:
            st.metric("Random Forest Risk", f"{cl_risk:.1f}%")
            st.progress(min(cl_risk / 100, 1.0))

        # QSVM prediction
        try:
            qsvm_pred = cached["qsvm"].predict_proba(X_q)[:, 1][0] * 100
        except Exception:
            qsvm_pred = 50.0
        with res_cols[1]:
            st.metric("Quantum Kernel Risk", f"{qsvm_pred:.1f}%")
            st.progress(min(qsvm_pred / 100, 1.0))

        # VQC prediction
        try:
            vqc_pred = cached["vqc"].predict(X_q)[0]
            vqc_label = "HIGH RISK" if vqc_pred == 1 else "LOW RISK"
        except Exception:
            vqc_label = "N/A"
        with res_cols[2]:
            st.metric("VQC Classification", vqc_label)

        # Risk summary
        avg_risk = (cl_risk + qsvm_pred) / 2
        if avg_risk > 60:
            st.error(f"⚠️ Combined Risk Score: {avg_risk:.1f}% — Further evaluation recommended")
        elif avg_risk > 30:
            st.warning(f"🔶 Combined Risk Score: {avg_risk:.1f}% — Monitor closely")
        else:
            st.success(f"✅ Combined Risk Score: {avg_risk:.1f}% — Low risk indicators")


def render_quantum_circuits(cached):
    """Display quantum circuit information."""
    st.header("⚛️ Quantum Circuit Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QSVM Feature Map")
        circuit_text = cached["qsvm"].get_circuit_diagram()
        if circuit_text:
            st.code(circuit_text, language="text")
        else:
            st.info("Quantum circuit diagram available when Qiskit is installed")

    with col2:
        st.subheader("VQC Full Circuit")
        vqc_text = cached["vqc"].get_circuit_diagram()
        if vqc_text:
            st.code(vqc_text, language="text")
        else:
            st.info("Quantum circuit diagram available when Qiskit is installed")

    # Training convergence
    history = cached.get("vqc_history", [])
    if history:
        st.subheader("📈 VQC Training Convergence")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color="#7C3AED", linewidth=2)
        ax.fill_between(range(len(history)), history, alpha=0.15, color="#7C3AED")
        ax.set_xlabel("Iteration", color="white")
        ax.set_ylabel("Loss", color="white")
        ax.set_title("VQC Optimization Convergence", fontweight="bold", color="white")
        ax.set_facecolor("#1a1040")
        fig.patch.set_facecolor("#0f0c29")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)


def render_dataset_overview(dataset):
    """Show dataset statistics."""
    st.header("📋 Dataset Overview")

    combined = dataset["combined"]
    labels = dataset["labels"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", combined.shape[0])
    col2.metric("Total Features", combined.shape[1])
    col3.metric("Positive Cases", int(labels.sum()))
    col4.metric("Prevalence", f"{labels.mean() * 100:.1f}%")

    st.subheader("Feature Distribution (First 6 Features)")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < combined.shape[1]:
            col = combined.columns[idx]
            ax.hist(
                combined[col][labels == 0], bins=25, alpha=0.6,
                label="Healthy", color="#10B981"
            )
            ax.hist(
                combined[col][labels == 1], bins=25, alpha=0.6,
                label="Affected", color="#EF4444"
            )
            ax.set_title(col, fontsize=9, color="white")
            ax.tick_params(colors="white", labelsize=7)
            ax.set_facecolor("#1a1040")
    axes[0, 0].legend(fontsize=7)
    fig.patch.set_facecolor("#0f0c29")
    plt.tight_layout()
    st.pyplot(fig)


def render_footer():
    """Render the dashboard footer."""
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#666; font-size:0.8rem;">'
        '⚛️ Q-Glitch | Team VTU-QFF-HC-0190 | '
        'VTU-Qiskit Fall Fest 2026<br/>'
        '© 2026 Visvesvaraya Technological University, Belagavi, '
        'in association with IBM Quantum.<br/>'
        'Qiskit and IBM Quantum are trademarks of IBM Corporation.'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
def main():
    apply_custom_css()
    render_header()

    # Sidebar navigation
    st.sidebar.markdown("## ⚛️ Q-Glitch Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Overview & Comparison", "🩺 Patient Assessment",
         "⚛️ Quantum Circuits", "📋 Dataset Explorer"]
    )

    # Load and train models (cached)
    with st.spinner("⚛️ Initializing quantum-classical pipeline..."):
        cached = load_and_train_models()

    if page == "🏠 Overview & Comparison":
        render_model_comparison(cached["results"])
    elif page == "🩺 Patient Assessment":
        render_patient_assessment(cached)
    elif page == "⚛️ Quantum Circuits":
        render_quantum_circuits(cached)
    elif page == "📋 Dataset Explorer":
        render_dataset_overview(cached["dataset"])

    render_footer()


if __name__ == "__main__":
    main()
