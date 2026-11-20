"""
Page 6: Probability Calibration, Reliability Diagrams, and Risk Quantification.
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
    load_calibration_data,
    get_calibration_summary_metrics,
    load_sample_cases,
)
from utils.visualizations import plot_reliability_curve

st.set_page_config(page_title="Calibration - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Probability Calibration & Reliability Analysis")
st.caption("Quantifying the Reliability of Estimated Risk Scores for Clinical Decision Support")

df_cal = load_calibration_data()
cal_summary = get_calibration_summary_metrics()

tab_reliability, tab_bins, tab_prevalence = st.tabs([
    "Reliability Diagrams & ECE",
    "10-Bin Calibration Diagnostics Table",
    "Base Rate & Population Prevalence Warning",
])

with tab_reliability:
    st.subheader("Reliability Curves: Observed Outcome vs Predicted Confidence")
    st.markdown(
        """
        A risk prediction model is **well-calibrated** if, among patients assigned a predicted risk $p$, 
        an empirical proportion $p$ actually have diabetes. In a reliability diagram, perfectly calibrated 
        probabilities fall directly along the dashed diagonal line ($y = x$).
        """
    )
    
    # KPI metrics
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric(
            "Model 1 Brier Score",
            f"{cal_summary['baseline_model']['brier_score']:.4f}",
            help="Mean squared error between predicted probability and true label (0/1). Lower is better.",
        )
    with c_m2:
        st.metric(
            "Model 2 Brier Score",
            f"{cal_summary['trustworthy_model']['brier_score']:.4f}",
            delta="+0.0001 (Negligible)",
            delta_color="inverse",
        )
    with c_m3:
        st.metric(
            "Model 1 ECE (10 Bins)",
            f"{cal_summary['baseline_model']['ece']:.4f}",
            help="Expected Calibration Error: Weighted average gap between mean confidence and accuracy. Lower is better.",
        )
    with c_m4:
        st.metric(
            "Model 2 ECE (10 Bins)",
            f"{cal_summary['trustworthy_model']['ece']:.4f}",
            delta="+0.0018 (Slightly Higher)",
            delta_color="inverse",
            help="Empirical finding: Platt calibration slightly increased ECE from 0.0262 to 0.0280.",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_reliability_curve(df_cal, cal_summary), use_container_width=True)

    st.markdown("---")
    st.subheader("Objective Scientific Finding on Probability Calibration")
    st.markdown(
        """
        In machine learning research, post-hoc calibration methods (Platt sigmoid scaling or Isotonic regression) 
        are frequently assumed to automatically improve probability accuracy. Our empirical findings demonstrate 
        why this assumption must be tested rather than taken for granted:
        
        1. **Logistic Regression is Naturally Calibrated under Binary Cross-Entropy:**  
           The negative log-likelihood (cross-entropy) loss function directly optimizes Bernoulli likelihood. 
           When test feature distributions and positive prevalence match the training cohort, Logistic Regression 
           produces inherently well-calibrated probabilities (Baseline ECE = **0.0262**).
           
        2. **Validation Sample Variance in Platt Scaling:**  
           Fitting a two-parameter sigmoid $\\frac{1}{1 + e^{A z + B}}$ on a finite validation set ($N = 10,604$) 
           inherits small estimation variance. Applying this secondary mapping to the test set resulted in an ECE 
           of **0.0280**—a slight increase (+0.0018) rather than an improvement.
           
        3. **Where Calibration Genuinely Matters:**  
           Post-hoc calibration is essential for algorithms whose raw outputs do not represent true log-odds 
           (e.g., Support Vector Machines, Random Forests, Naive Bayes with feature dependencies, or Neural Networks 
           subject to overconfidence).
        """
    )

with tab_bins:
    st.subheader("Detailed 10-Bin Calibration Diagnostics Table")
    st.write("Examine the mean predicted probability vs observed fraction of positives across decile probability bins:")
    
    df_table = df_cal.copy()
    df_table["Baseline Bin Gap (|Pred - Frac|)"] = (df_table["baseline_mean_pred"] - df_table["baseline_frac_pos"]).abs().round(4)
    df_table["Trustworthy Bin Gap (|Pred - Frac|)"] = (df_table["trustworthy_mean_pred"] - df_table["trustworthy_frac_pos"]).abs().round(4)
    
    df_table = df_table.rename(columns={
        "bin_index": "Decile Bin",
        "baseline_mean_pred": "Model 1 Mean Pred (x)",
        "baseline_frac_pos": "Model 1 Observed Frac (y)",
        "trustworthy_mean_pred": "Model 2 Mean Pred (x)",
        "trustworthy_frac_pos": "Model 2 Observed Frac (y)",
    })
    
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    st.info(
        "**Decile Analysis:**  \n"
        "• At low confidence (Bin 1: ~6% predicted risk), the observed fraction is ~2.8% (slight overestimation).  \n"
        "• At medium confidence (Bins 4-7: 35% - 65% predicted risk), the model closely tracks empirical reality (error < 3%).  \n"
        "• At very high confidence (Bin 10: ~93% predicted risk), the observed fraction is ~87.6% (slight overconfidence)."
    )

with tab_prevalence:
    st.subheader("Critical Clinical Warning: The Impact of 50/50 Balanced Sampling")
    st.markdown(
        """
        <div class="disclaimer-banner">
            <h4 style="color: #991b1b; margin-top: 0;">Clinical Decision Support Notice: Probability Translation</h4>
            <p>
            The CDC BRFSS 2015 dataset used here was synthetically balanced to an exact <b>50% diabetes / 50% non-diabetic</b> cohort.
            In real-world primary care or population screening in the United States, the prevalence of diagnosed diabetes is approximately <b>10% to 14%</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        #### Bayes' Rule and Prior Probability Shifts:
        When a classifier is trained on an artificially balanced 50/50 dataset, its predicted posterior probability 
        $P(Y=1 \\mid x)$ reflects a prior odds of $1:1$ (odds = 1.0).
        
        In a clinical population with a true prevalence $\\pi_0 \\approx 0.12$ (odds $\\approx 0.136$), 
        the true positive predictive value (PPV) will be **substantially lower** than the raw probabilities output 
        by this model.
        
        To convert the model's predicted probability $P_{\\text{model}}$ to a real-world population probability 
        $P_{\\text{real}}$ under a true prevalence $\\pi_{\\text{real}}$, Bayes' rule applies:
        $$\\text{Odds}_{\\text{real}} = \\text{Odds}_{\\text{model}} \\times \\frac{\\pi_{\\text{real}}}{1 - \\pi_{\\text{real}}} \\times \\frac{1 - \\pi_{\\text{model}}}{\\pi_{\\text{model}}}$$
        $$P_{\\text{real}} = \\frac{\\text{Odds}_{\\text{real}}}{1 + \\text{Odds}_{\\text{real}}}$$
        
        *Example:* A patient assigned a 50% probability in this 50/50 dataset would have an estimated real-world probability 
        of approximately **12% to 14%** in the general adult US population.
        """
    )
