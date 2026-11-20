"""
Page 2: Rigorous Model Performance & Discrimination Comparison.
"""

import sys
import os

# Ensure application directory and project root are on sys.path
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import apply_custom_css
from utils.data_loader import (
    load_evaluation_metrics,
    get_calibration_summary_metrics,
)
from utils.visualizations import plot_confusion_matrix, plot_roc_curves

st.set_page_config(page_title="Model Comparison - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Comprehensive Model Comparison")
st.caption("Discrimination vs Calibration: Evaluating Model 1 (Baseline LR) vs Model 2 (Trust-Aware Calibrated LR)")

metrics = load_evaluation_metrics()
cal_summary = get_calibration_summary_metrics()

m1 = metrics["baseline_model"]
m2 = metrics["trustworthy_model"]

# Section 1: Side-by-side Metric Table & KPI Cards
st.subheader("1. Test Set Performance Metrics (N = 10,604)")

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("ROC-AUC", f"{m1['roc_auc']:.4f}", delta="Identical (0.8251)")
with k2:
    st.metric("Accuracy", f"{m1['accuracy']:.2%}", delta=f"M2: {m2['accuracy']:.2%}")
with k3:
    st.metric("Sensitivity (Recall)", f"{m1['recall']:.2%}", delta=f"M2: {m2['recall']:.2%}")
with k4:
    st.metric("Precision (PPV)", f"{m1['precision']:.2%}", delta=f"M2: {m2['precision']:.2%}")
with k5:
    st.metric("F1-Score", f"{m1['f1']:.4f}", delta=f"M2: {m2['f1']:.4f}")
with k6:
    st.metric("Brier Score", f"{m1['brier_score']:.4f}", delta=f"M2: {m2['brier_score']:.4f}", delta_color="inverse")

# Full Comparison Table
df_comp = pd.DataFrame([
    {
        "Evaluation Dimension": "Discrimination (Ranking)",
        "Metric": "ROC-AUC",
        "Model 1 (Baseline LR)": f"{m1['roc_auc']:.4f}",
        "Model 2 (Calibrated LR)": f"{m2['roc_auc']:.4f}",
        "Difference (M2 - M1)": f"{m2['roc_auc'] - m1['roc_auc']:+.4f}",
        "Scientific Interpretation": "Identical ranking. Platt scaling is strictly monotonic.",
    },
    {
        "Evaluation Dimension": "Classification Utility",
        "Metric": "Overall Accuracy",
        "Model 1 (Baseline LR)": f"{m1['accuracy']:.4f} ({m1['accuracy']:.2%})",
        "Model 2 (Calibrated LR)": f"{m2['accuracy']:.4f} ({m2['accuracy']:.2%})",
        "Difference (M2 - M1)": f"{m2['accuracy'] - m1['accuracy']:+.4f}",
        "Scientific Interpretation": "Practically identical utility across binary decision cutoffs.",
    },
    {
        "Evaluation Dimension": "Screening Sensitivity",
        "Metric": "True Positive Rate (Recall)",
        "Model 1 (Baseline LR)": f"{m1['recall']:.4f} ({m1['recall']:.2%})",
        "Model 2 (Calibrated LR)": f"{m2['recall']:.4f} ({m2['recall']:.2%})",
        "Difference (M2 - M1)": f"{m2['recall'] - m1['recall']:+.4f}",
        "Scientific Interpretation": "Baseline flags slightly more true positives (+33 patients).",
    },
    {
        "Evaluation Dimension": "Screening Precision",
        "Metric": "Precision (PPV)",
        "Model 1 (Baseline LR)": f"{m1['precision']:.4f} ({m1['precision']:.2%})",
        "Model 2 (Calibrated LR)": f"{m2['precision']:.4f} ({m2['precision']:.2%})",
        "Difference (M2 - M1)": f"{m2['precision'] - m1['precision']:+.4f}",
        "Scientific Interpretation": "Calibrated model produces slightly fewer false alarms.",
    },
    {
        "Evaluation Dimension": "Harmonic Utility",
        "Metric": "F1-Score",
        "Model 1 (Baseline LR)": f"{m1['f1']:.4f}",
        "Model 2 (Calibrated LR)": f"{m2['f1']:.4f}",
        "Difference (M2 - M1)": f"{m2['f1'] - m1['f1']:+.4f}",
        "Scientific Interpretation": "Negligible difference (-0.0016).",
    },
    {
        "Evaluation Dimension": "Probability Reliability",
        "Metric": "Brier Score (MSE)",
        "Model 1 (Baseline LR)": f"{m1['brier_score']:.5f}",
        "Model 2 (Calibrated LR)": f"{m2['brier_score']:.5f}",
        "Difference (M2 - M1)": f"{m2['brier_score'] - m1['brier_score']:+.5f}",
        "Scientific Interpretation": "Nearly identical mean squared error in probabilities.",
    },
    {
        "Evaluation Dimension": "Probability Reliability",
        "Metric": "Expected Calibration Error (ECE)",
        "Model 1 (Baseline LR)": f"{cal_summary['baseline_model']['ece']:.4f}",
        "Model 2 (Calibrated LR)": f"{cal_summary['trustworthy_model']['ece']:.4f}",
        "Difference (M2 - M1)": f"{cal_summary['trustworthy_model']['ece'] - cal_summary['baseline_model']['ece']:+.4f}",
        "Scientific Interpretation": "Baseline has slightly better ECE (0.0262 vs 0.0280).",
    },
])

st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 2: Overlaid ROC Curves & Confusion Matrices
st.subheader("2. Discrimination Diagnostics: Overlaid ROC & Confusion Matrices")

col_roc, col_cm = st.columns([1.1, 1.0])

with col_roc:
    st.plotly_chart(plot_roc_curves(metrics), use_container_width=True)

with col_cm:
    st.write("**Side-by-Side Confusion Matrices (Threshold = 0.50, N = 10,604)**")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_confusion_matrix(m1["confusion_matrix"], "Model 1 (Baseline)"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_confusion_matrix(m2["confusion_matrix"], "Model 2 (Trust-Aware)"), use_container_width=True)

st.markdown("---")

# Section 3: Scientific Discussion
st.subheader("3. Scientific Principles & Methodological Nuances")

st.markdown(
    """
    #### A. Why Are the ROC Curves and AUC Values Exactly Identical?
    The Receiver Operating Characteristic (ROC) curve evaluates a classifier's ability to rank positive 
    instances above negative instances across all possible decision thresholds.
    
    Model 2 wraps Model 1's fitted pipeline in `CalibratedClassifierCV` using Platt sigmoid calibration. 
    Platt scaling maps the inner model's continuous decision score $s(x)$ through a univariate logistic function:
    $$P(Y=1 \\mid s) = \\frac{1}{1 + \\exp(A \\cdot s(x) + B)}$$
    Because the learned scale parameter $A < 0$ and the exponential function is strictly monotonic, 
    the transformation preserves the exact ordinal ranking of all 10,604 test instances:
    $$s(x_i) > s(x_j) \\iff P(Y=1 \\mid s(x_i)) > P(Y=1 \\mid s(x_j))$$
    Therefore, the False Positive Rate (FPR) and True Positive Rate (TPR) remain identical at every rank quantile, 
    resulting in mathematically **identical ROC curves and AUC (0.8251)**.

    #### B. Why Did Calibration Not Improve ECE on this Dataset?
    In standard Logistic Regression with log-loss (cross-entropy), the optimization objective naturally encourages 
    calibrated probabilities on the training distribution. Because the validation split and test split share the 
    exact same 50/50 balance and feature distributions as the training set, Model 1 was already exceptionally 
    well-calibrated (ECE = 0.0262). Fitting an additional 2-parameter sigmoid on the validation set introduces 
    minor estimation variance, resulting in an ECE of 0.0280.
    
    *Conclusion:* Calibration is not an automatic 'performance booster'; its value lies in verifying and correcting 
    probability distortions when models are non-probabilistic (e.g., SVMs) or deployed under distribution shift.
    """
)
