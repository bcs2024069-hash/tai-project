"""
Page 4: Global and Individual SHAP Explainability & Attribution Analysis.
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
from utils.helpers import (
    apply_custom_css,
    FEATURE_METADATA,
    format_feature_value,
)
from utils.data_loader import (
    load_shap_global,
    load_shap_individual,
    load_sample_cases,
)
from utils.visualizations import (
    plot_global_shap,
    plot_local_shap_horizontal,
)

st.set_page_config(page_title="SHAP Explainability - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Model Explainability via SHAP (SHapley Additive exPlanations)")
st.caption("Global Risk Drivers, Local Patient Attributions, and the Log-Odds Mathematical Link")

df_global = load_shap_global()
df_individual = load_shap_individual()
df_cases = load_sample_cases()

tab_global, tab_local, tab_theory = st.tabs([
    "Global Feature Importance",
    "Local Patient Explanations (15 Cases)",
    "Log-Odds vs Calibrated Probability Theory",
])

with tab_global:
    st.subheader("Global Risk Drivers Across the Entire Cohort")
    st.markdown(
        """
        Global SHAP importance measures the average absolute contribution of each feature to the model's 
        log-odds decision score across 1,000 background test respondents:
        $$\\text{Global Importance}(j) = \\frac{1}{N} \\sum_{i=1}^N |\\phi_j(x_i)|$$
        """
    )
    
    col_chart, col_table = st.columns([1.3, 1.0])
    with col_chart:
        top_n = st.slider("Select Number of Top Features to Display:", 5, 21, 15)
        st.plotly_chart(plot_global_shap(df_global, top_n=top_n), use_container_width=True)
        
    with col_table:
        st.write("**Ranked Feature Importance Table:**")
        df_display_g = df_global.copy()
        df_display_g["Clinical Name"] = df_display_g["feature"].map(lambda x: FEATURE_METADATA.get(x, {}).get("name", x))
        df_display_g["Domain"] = df_display_g["feature"].map(lambda x: FEATURE_METADATA.get(x, {}).get("category", "General"))
        st.dataframe(
            df_display_g[["rank", "Clinical Name", "feature", "mean_abs_shap", "Domain"]],
            use_container_width=True,
            hide_index=True,
            height=430,
        )
        
    st.info(
        "**Clinical Interpretation:**  \n"
        "• **General Health (`GenHlth` - Rank 1, Mean |SHAP| = 0.5072):** The single strongest predictor. Patients rating their health as Poor (5) or Fair (4) experience the largest upward push toward diabetes risk.  \n"
        "• **Body Mass Index (`BMI` - Rank 2, Mean |SHAP| = 0.4104):** Key physiological indicator reflecting metabolic adiposity.  \n"
        "• **High Blood Pressure (`HighBP` - Rank 3, Mean |SHAP| = 0.3513) & High Cholesterol (`HighChol` - Rank 5, Mean |SHAP| = 0.2924):** Hallmarks of metabolic syndrome.  \n"
        "• **Age (`Age` - Rank 4, Mean |SHAP| = 0.3301):** Strongly monotonic age risk gradient, reflecting increasing insulin resistance over lifespan."
    )

with tab_local:
    st.subheader("Local Patient Feature Attributions (Case-Level)")
    st.markdown(
        """
        Select one of the 15 curated patient cases to inspect how each specific health indicator 
        pushed the model toward or away from a positive diabetes screening score:
        """
    )
    
    col_case_sel, col_case_info = st.columns([1, 2])
    with col_case_sel:
        case_id = st.selectbox("Select Case to Inspect:", df_individual["case_id"].tolist())
    
    case_row = df_cases[df_cases["case_id"] == case_id].iloc[0]
    shap_row = df_individual[df_individual["case_id"] == case_id].iloc[0]
    
    with col_case_info:
        actual_lbl = "Diabetic / Pre-diabetic" if case_row["actual_label"] == 1 else "Non-Diabetic"
        st.markdown(
            f"""
            **Case: `{case_id}`**  
            • **Actual Outcome:** {actual_lbl} | **Model 1 Prob:** `{case_row['model1_prob']:.1%}` | **Model 2 Prob:** `{case_row['model2_prob']:.1%}`  
            • **Patient Profile:** Age: `{format_feature_value('Age', case_row['Age'])}` | BMI: `{case_row['BMI']:.1f}` | GenHlth: `{format_feature_value('GenHlth', case_row['GenHlth'])}`
            """
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_waterfall, col_feat_vals = st.columns([1.3, 1.0])
    with col_waterfall:
        max_feats = st.slider("Max Features to Display:", 5, 21, 10, key="local_feat_slider")
        st.plotly_chart(plot_local_shap_horizontal(shap_row, max_display=max_feats), use_container_width=True)
        
    with col_feat_vals:
        st.write("**Patient Feature Values & Exact Attributions:**")
        local_table = []
        for feat in [c for c in shap_row.index if c != "case_id"]:
            val = case_row.get(feat, None)
            local_table.append({
                "Feature": feat,
                "Value": format_feature_value(feat, val),
                "SHAP Impact (Log-Odds)": f"{shap_row[feat]:+.4f}",
                "Direction": "Increases Risk" if shap_row[feat] > 0 else "Decreases Risk",
            })
        df_loc_tab = pd.DataFrame(local_table).sort_values("SHAP Impact (Log-Odds)", ascending=False)
        st.dataframe(df_loc_tab, use_container_width=True, hide_index=True, height=380)

with tab_theory:
    st.subheader("Mathematical Foundations: Log-Odds vs Calibrated Probability")
    st.markdown(
        """
        A critical consideration in Trustworthy AI is understanding **which mathematical space** explainability 
        methods operate in.
        
        #### 1. The Logistic Regression Scoring Function
        In the standard Logistic Regression model (Model 1), the linear combination of standardized features 
        computes an unconstrained continuous decision score $z(x)$, known as the **log-odds** or **logit**:
        $$z(x) = \\ln \\left( \\frac{P(Y=1 \\mid x)}{1 - P(Y=1 \\mid x)} \\right) = \\beta_0 + \\sum_{j=1}^p \\beta_j \\tilde{x}_j$$
        where $\\tilde{x}_j$ represents the standardized or passthrough feature value.
        
        #### 2. Why SHAP LinearExplainer Operates on Log-Odds
        By the Shapley additive attribution theorem, the sum of feature attributions $\\phi_j(x)$ plus the base value $\\phi_0$ 
        must sum exactly to the model output:
        $$z(x) = \\phi_0 + \\sum_{j=1}^p \\phi_j(x)$$
        Because linear models are additive in **log-odds space**, `shap.LinearExplainer` computes exact closed-form 
        attributions:
        $$\\phi_j(x) = \\beta_j \\cdot (\\tilde{x}_j - \\mathbb{E}[\\tilde{x}_j])$$
        
        If SHAP were computed in probability space, the non-linear sigmoid link $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ 
        would induce non-additive cross-feature interaction terms, violating efficiency and additivity.
        
        #### 3. How SHAP Values Relate to Model 2's Calibrated Probability
        Model 2 wraps the inner linear score $z(x)$ in Platt sigmoid calibration:
        $$P_{\\text{cal}}(Y=1 \\mid z) = \\frac{1}{1 + \\exp(A \\cdot z(x) + B)}$$
        
        Because Platt scaling is a strictly monotonic 1D transformation of $z(x)$, a feature that increases the log-odds 
        ($\\phi_j > 0$) strictly increases both:
        1. The uncalibrated probability $\\sigma(z(x))$.
        2. The calibrated probability $P_{\\text{cal}}(z(x))$.
        
        Therefore, **SHAP values on the inner model remain faithful qualitative explanations of the calibrated pipeline's predictions**, 
        while the exact numerical SHAP magnitude represents log-odds impact.
        """
    )
