"""
Trustworthy Diabetes Risk-Screening Decision System
Main Streamlit Application Entrypoint & Executive Overview.
"""

import sys
import os

# Ensure application directory and project root are on sys.path
APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from utils.helpers import apply_custom_css
from utils.data_loader import (
    load_metadata,
    load_evaluation_metrics,
    get_calibration_summary_metrics,
)

# Page configuration
st.set_page_config(
    page_title="Trustworthy Diabetes Risk-Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Header
st.title("Trustworthy Diabetes Risk-Screening Decision System")
st.caption(
    "An Academic Comparative Evaluation of Standard vs Trust-Aware Machine Learning "
    "on CDC BRFSS 2015 Health Survey Data"
)

# Executive Summary
st.markdown("---")
st.subheader("Executive Summary & Research Motivation")

st.markdown(
    """
    In population health and chronic disease prevention, machine learning models frequently support 
    initial risk screening. However, conventional predictive pipelines emphasize raw discriminatory 
    metrics (e.g., ROC-AUC, Accuracy) while neglecting **trustworthiness pillars**:
    - **Probability Reliability:** Are estimated risks mathematically aligned with empirical outcome frequencies?
    - **Algorithmic Fairness:** Do false positive or false negative rates disproportionately penalize demographic subgroups?
    - **Feature Explainability:** Are predictions interpretable and clinically grounded at both global and individual levels?
    - **Input Robustness:** Are predictions stable under slight, plausible measurement noise or self-reporting variation?

    This interactive system implements and rigorously compares two predictive pipelines trained on the 
    **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** dataset (70,692 respondents, 21 survey features):
    """
)

# Two Model Architecture Cards
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown(
        """
        <div style="background-color: #f8fafc; border: 2px solid #cbd5e1; border-radius: 8px; padding: 18px;">
            <h4 style="color: #1e40af; margin-top: 0;">Model 1: Standard Logistic Regression Baseline</h4>
            <p><b>Architecture:</b> Standardized continuous/ordinal features + passthrough binary indicators fed to an L2-regularized Logistic Regression classifier (L-BFGS solver).</p>
            <ul>
                <li>Standard 0.50 decision threshold.</li>
                <li>Uncalibrated probability estimation via raw sigmoid link.</li>
                <li>Standard statistical performance metrics only.</li>
                <li>No embedded trustworthiness or fairness evaluation layer.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        """
        <div style="background-color: #f0fdf4; border: 2px solid #86efac; border-radius: 8px; padding: 18px;">
            <h4 style="color: #166534; margin-top: 0;">Model 2: Trust-Aware Decision Framework</h4>
            <p><b>Architecture:</b> Identical underlying Logistic Regression pipeline wrapped with <code>FrozenEstimator</code> and post-hoc sigmoid calibration fitted strictly on validation data.</p>
            <ul>
                <li>Post-hoc probability calibration (Platt sigmoid scaling).</li>
                <li>Model explainability via SHAP (LinearExplainer on log-odds).</li>
                <li>Fairlearn subgroup performance auditing (Sex, Age, Income).</li>
                <li>Input robustness testing across 6 controlled survey perturbations.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Key Global Metrics
metrics = load_evaluation_metrics()
cal_summary = get_calibration_summary_metrics()
m1_data = metrics["baseline_model"]
m2_data = metrics["trustworthy_model"]

st.subheader("Key Performance & Reliability Summary (Test Set, N = 10,604)")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="ROC-AUC (Both Models)",
        value=f"{m1_data['roc_auc']:.4f}",
        delta="Identical (0.8251)",
        help="Platt calibration is a strictly monotonic sigmoid transformation of log-odds, preserving exact ranking.",
    )

with kpi2:
    st.metric(
        label="Model 1 Accuracy",
        value=f"{m1_data['accuracy']:.2%}",
        delta=f"Brier: {m1_data['brier_score']:.4f}",
    )

with kpi3:
    st.metric(
        label="Model 2 Accuracy",
        value=f"{m2_data['accuracy']:.2%}",
        delta=f"Brier: {m2_data['brier_score']:.4f}",
    )

with kpi4:
    st.metric(
        label="Model 1 ECE",
        value=f"{cal_summary['baseline_model']['ece']:.4f}",
        help="Expected Calibration Error (10 bins). Lower is better.",
    )

with kpi5:
    st.metric(
        label="Model 2 ECE",
        value=f"{cal_summary['trustworthy_model']['ece']:.4f}",
        delta="Slightly higher (+0.0018)",
        delta_color="inverse",
        help="Calibration did not improve ECE on this balanced dataset, demonstrating empirical objectivity.",
    )

# Page Navigation Guide
st.markdown("---")
st.subheader("Explore the System Architecture")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### [1. Project Overview](Project_Overview)")
    st.write("Examine the CDC BRFSS 2015 cohort, 21-feature codebook, data splitting methodology, and duplicate survey analysis.")
    
    st.markdown("### [2. Model Comparison](Model_Comparison)")
    st.write("Compare side-by-side performance metrics, overlaid ROC curves, and confusion matrices with scientific rigor.")

with c2:
    st.markdown("### [3. Case Predictions](Case_Predictions)")
    st.write("Inspect 15 curated patient profiles with side-by-side risk predictions, true labels, and risk factor summaries.")
    
    st.markdown("### [4. SHAP Explainability](SHAP_Explainability)")
    st.write("Explore global feature importance rankings and local SHAP waterfall attributions explaining underlying log-odds.")

with c3:
    st.markdown("### [5. Fairness Audit](Fairness_Audit)")
    st.write("Audit demographic parity and equalized odds disparities across Sex, Age, and Income, plus mitigation trade-offs.")
    
    st.markdown("### [6. Calibration](Calibration) & [7. Robustness](Robustness)")
    st.write("Analyze reliability diagrams, ECE scores, and stability under 6 controlled feature perturbations.")

st.sidebar.title("Navigation")
st.sidebar.info("Select any page from the sidebar navigation menu above to inspect detailed evaluations.")
st.sidebar.markdown("---")
st.sidebar.caption("Trustworthy Diabetes Screening System • Academic Prototype • v1.0")
