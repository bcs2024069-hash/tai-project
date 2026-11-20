"""
Page 7: Input Robustness & Controlled Perturbation Sensitivity Stress Testing.
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
from utils.helpers import apply_custom_css
from utils.data_loader import (
    load_robustness_summary,
    load_robustness_results,
)
from utils.visualizations import plot_robustness_shifts

st.set_page_config(page_title="Robustness - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Input Robustness & Controlled Perturbation Sensitivity")
st.caption("Stress Testing Stability under Plausible Self-Reported Survey Noise Across 500 Test Cases")

df_summary = load_robustness_summary()
df_results = load_robustness_results()

tab_summary, tab_explorer, tab_methodology = st.tabs([
    "Perturbation Impact Summary",
    "Sample Case Level Results Explorer",
    "Sensitivity vs Adversarial Robustness",
])

PERTURBATION_MAP = {
    "bmi_plus1": "BMI (+1 kg/m²)",
    "bmi_minus1": "BMI (-1 kg/m²)",
    "genhlth_worse": "General Health (+1 rating category worse)",
    "genhlth_better": "General Health (-1 rating category better)",
    "physhlth_plus1": "Physical Illness Days (+1 day)",
    "menthlth_plus1": "Mental Stress Days (+1 day)",
}

with tab_summary:
    st.subheader("Comparative Perturbation Stress Test Results")
    st.markdown(
        """
        To evaluate model sensitivity under plausible survey reporting variations or minor measurement noise, 
        we executed **6 controlled single-feature perturbations** across 500 randomly sampled test cases, 
        strictly respecting the BRFSS codebook valid ranges:
        """
    )
    
    st.plotly_chart(plot_robustness_shifts(df_summary), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Aggregated Robustness Summary Table")
    
    df_disp = df_summary.copy()
    df_disp["Perturbation Description"] = df_disp["perturbation"].map(PERTURBATION_MAP)
    df_disp["Mean Shift (|Δp|)"] = df_disp["mean_abs_prob_change"].map(lambda x: f"{x:.4f}")
    df_disp["Max Shift (|Δp|)"] = df_disp["max_abs_prob_change"].map(lambda x: f"{x:.4f}")
    df_disp["Prediction Flip Rate"] = df_disp["flip_rate"].map(lambda x: f"{x:.2%}")
    df_disp["Flips / Total (N=500)"] = df_disp.apply(lambda r: f"{r['flip_count']} / {r['n_samples']}", axis=1)
    
    cols = ["model", "Perturbation Description", "Mean Shift (|Δp|)", "Max Shift (|Δp|)", "Prediction Flip Rate", "Flips / Total (N=500)"]
    st.dataframe(df_disp[cols], use_container_width=True, hide_index=True)
    
    st.info(
        "**Key Robustness Findings:**  \n"
        "• **General Health Dominates (`genhlth_worse` / `genhlth_better`):** Moving self-reported general health by just one category (e.g., from 'Good' to 'Fair') alters predicted probability by an average of **~7.8%**, flipping **12.6% to 14.2%** of all binary decisions at the 0.50 threshold. This aligns with `GenHlth` being the top SHAP predictor.  \n"
        "• **Moderate BMI Impact:** A 1-unit change in BMI (~6-7 lbs for an average adult) alters probability by **~1.4%**, causing **1.0% to 1.8%** of borderline predictions to flip.  \n"
        "• **High Stability on Mental & Physical Health Days:** Adding 1 day of physical or mental illness causes negligible change (<0.3%), flipping only **1 to 3 cases out of 500**."
    )

with tab_explorer:
    st.subheader("Sample-Level Perturbation Results Explorer")
    st.write("Inspect how individual test patient probabilities and decisions shifted under specific perturbations:")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_pert = st.selectbox(
            "Filter by Perturbation Scenario:",
            list(PERTURBATION_MAP.keys()),
            format_func=lambda x: PERTURBATION_MAP[x],
        )
    with col_f2:
        sel_model = st.selectbox("Filter by Model:", ["Baseline", "Trust-Aware"])
    with col_f3:
        sel_flipped = st.selectbox("Filter by Prediction Flip:", ["All Cases", "Flipped Predictions Only", "Non-Flipped Predictions Only"])
        
    filtered_df = df_results[(df_results["perturbation"] == sel_pert) & (df_results["model"] == sel_model)].copy()
    
    if sel_flipped == "Flipped Predictions Only":
        filtered_df = filtered_df[filtered_df["flipped"] == 1]
    elif sel_flipped == "Non-Flipped Predictions Only":
        filtered_df = filtered_df[filtered_df["flipped"] == 0]
        
    st.write(f"Displaying **{len(filtered_df)}** sample records:")
    
    disp_res = filtered_df.copy()
    disp_res["Original Prob"] = disp_res["original_prob"].map(lambda x: f"{x:.1%}")
    disp_res["Perturbed Prob"] = disp_res["perturbed_prob"].map(lambda x: f"{x:.1%}")
    disp_res["Absolute Shift |Δp|"] = disp_res["abs_prob_change"].map(lambda x: f"{x:.4f}")
    disp_res["Flipped Decision?"] = disp_res["flipped"].map(lambda x: "FLIPPED" if x == 1 else "Stable")
    
    st.dataframe(
        disp_res[["sample_index", "Original Prob", "Perturbed Prob", "Absolute Shift |Δp|", "Flipped Decision?"]],
        use_container_width=True,
        hide_index=True,
        height=380,
    )

with tab_methodology:
    st.subheader("Scientific Rigor: Sensitivity Testing vs Adversarial Robustness")
    st.markdown(
        r"""
        In literature and industry presentations, the term **'robustness'** is often overloaded. 
        It is vital to distinguish between two fundamentally different paradigms:
        
        #### 1. Adversarial Robustness (Security / Attack Vector)
        In computer vision and cybersecurity, adversarial robustness evaluates resilience against an active, 
        malicious adversary who computes gradient-directed, imperceptible perturbations (e.g., FGSM, PGD) 
        specifically engineered to fool the classifier.
        
        *Why that does not apply here:* In public health risk screening via telephone surveys (BRFSS), 
        survey respondents do not compute mathematical gradients to deceive the model.
        
        #### 2. Controlled Input Sensitivity (Clinical Plausibility Testing)
        In clinical risk screening, robustness means **stability under plausible self-reporting variability**:
        - A respondent might weigh 1 kg more or less depending on clothing or time of day ($\text{BMI} \pm 1$).
        - A respondent feeling slightly fatigued might answer 'Fair' instead of 'Good' on general health.
        - A respondent might misremember whether they had 3 or 4 days of joint pain in the last month.
        
        **Our Methodological Safeguards:**
        1. **Domain-Bound Clamping:** Perturbations never exceed valid survey codebook boundaries (e.g., `GenHlth` is strictly clamped between 1 and 5; `PhysHlth` is clamped between 0 and 30).
        2. **Clinically Plausible Magnitudes:** Only single, realistic 1-unit shifts are applied.
        3. **Transparent Reporting:** We explicitly report both the continuous probability shift $|\Delta p|$ and the categorical decision flip rate at threshold 0.50.
        """
    )
