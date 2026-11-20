"""
Page 5: Subgroup Fairness Auditing & Fairlearn Post-Processing Mitigation.
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
from utils.data_loader import load_fairness_metrics, load_metadata
from utils.visualizations import plot_subgroup_fairness, plot_mitigation_comparison

st.set_page_config(page_title="Fairness Audit - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Algorithmic Fairness & Demographic Subgroup Auditing")
st.caption("Evaluating Disparities Across Sex, Age, and Income with Post-Processing Mitigation Analysis")

fair_data = load_fairness_metrics()
meta = load_metadata()

tab_audit, tab_mitigation, tab_concepts = st.tabs([
    "Subgroup Performance Audit",
    "Fairness Mitigation (ThresholdOptimizer)",
    "Fairness Metrics & Clinical Definitions",
])

with tab_audit:
    st.subheader("Demographic Subgroup Performance Breakdown")
    st.write("Examine selection rates, sensitivity (TPR), and false alarm rates (FPR) across demographic dimensions:")
    
    # Select attribute and model
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        attr = st.selectbox("Select Demographic / Socioeconomic Attribute:", ["Sex", "Age", "Income"])
    with col_sel2:
        model_key = st.radio("Select Model to Inspect:", ["baseline_model", "trustworthy_model"], format_func=lambda x: "Model 1 (Baseline LR)" if "baseline" in x else "Model 2 (Calibrated LR)", horizontal=True)

    attr_info = fair_data[model_key]["attributes"][attr]
    dp_diff = attr_info["demographic_parity_difference"]
    eo_diff = attr_info["equalized_odds_difference"]
    groups = attr_info["group_metrics"]

    # Metrics Summary Cards
    m_c1, m_c2, m_c3 = st.columns(3)
    with m_c1:
        st.metric(
            "Demographic Parity Difference",
            f"{dp_diff:.4f}",
            help="Difference between highest and lowest selection rate across subgroups. 0 indicates perfect parity.",
        )
    with m_c2:
        st.metric(
            "Equalized Odds Difference",
            f"{eo_diff:.4f}",
            help="Maximum difference in either True Positive Rate or False Positive Rate between any two subgroups. 0 indicates perfect equalized odds.",
        )
    with m_c3:
        st.metric("Number of Evaluated Groups", f"{len(groups)} Subgroups")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Grouped Bar Chart
    metric_options = st.multiselect(
        "Select Metrics to Plot on Chart:",
        ["tpr", "fpr", "fnr", "selection_rate", "accuracy"],
        default=["tpr", "fpr", "selection_rate"],
        format_func=lambda x: {
            "tpr": "True Positive Rate (TPR / Recall)",
            "fpr": "False Positive Rate (FPR)",
            "fnr": "False Negative Rate (FNR / Miss Rate)",
            "selection_rate": "Selection Rate (Predicted Positive %)",
            "accuracy": "Accuracy",
        }[x],
    )
    
    if metric_options:
        st.plotly_chart(plot_subgroup_fairness(groups, attr, metric_options), use_container_width=True)
    else:
        st.warning("Please select at least one metric to display on the chart.")

    # Detailed Subgroup Data Table
    st.write(f"**Detailed Auditing Table: {attr} Subgroups (N = 10,604 Test Records)**")
    df_group_table = pd.DataFrame(groups)
    df_group_table = df_group_table.rename(columns={
        "group_name": "Subgroup",
        "n": "Sample Size (N)",
        "selection_rate": "Selection Rate",
        "tpr": "True Positive Rate (TPR)",
        "fpr": "False Positive Rate (FPR)",
        "fnr": "False Negative Rate (FNR)",
        "accuracy": "Accuracy",
        "mean_predicted_prob": "Mean Predicted Prob",
    })
    
    # Format percentages
    for pct_col in ["Selection Rate", "True Positive Rate (TPR)", "False Positive Rate (FPR)", "False Negative Rate (FNR)", "Accuracy", "Mean Predicted Prob"]:
        df_group_table[pct_col] = df_group_table[pct_col].map(lambda x: f"{x:.2%}")
        
    cols_order = ["group_code", "Subgroup", "Sample Size (N)", "Selection Rate", "True Positive Rate (TPR)", "False Positive Rate (FPR)", "False Negative Rate (FNR)", "Accuracy", "Mean Predicted Prob"]
    st.dataframe(df_group_table[cols_order], use_container_width=True, hide_index=True)

    # Clinical & Ethical Insights
    st.markdown("---")
    st.subheader(f"Clinical & Algorithmic Observations for {attr}")
    if attr == "Sex":
        st.markdown(
            """
            • **Equitable Calibration:** Model behavior is remarkably balanced between biological females and males.  
            • **Selection Rate:** Females 49.23% vs Males 54.95% (Demographic Parity Difference = 0.0572).  
            • **Sensitivity (TPR):** Females 75.90% vs Males 77.49% (Equalized Odds Difference = 0.0512).  
            • *Takeaway:* Biological sex does not exhibit severe algorithmic bias in this screening model.
            """
        )
    elif attr == "Age":
        st.markdown(
            """
            • **Large Selection Rate Disparity (0.7387):** The model flags only **2.94%** of young adults (age 18-24) as positive, but flags **76.81%** of older adults (age 80+).  
            • **Underlying Epidemiology:** Type 2 diabetes prevalence naturally surges with age due to progressive beta-cell dysfunction, cumulative adiposity, and decreased insulin sensitivity.  
            • **The Equalized Odds Dilemma (0.7131):** The model's True Positive Rate in the 18-24 cohort is only **14.29%** (miss rate 85.71%), whereas in the 80+ cohort it reaches **85.60%**.  
            • *Clinical Implication:* Young adults with early-onset diabetes are frequently under-screened by models relying heavily on self-reported risk factors.
            """
        )
    elif attr == "Income":
        st.markdown(
            """
            • **Socioeconomic Gradient (0.5125 Parity Diff):** Respondents earning <$10k/year have a selection rate of **72.02%**, compared to only **20.77%** for those earning ≥$75k/year.  
            • **Equalized Odds Difference (0.4307):** False Positive Rate is 45.69% for <$10k earners, versus only 7.74% for high earners.  
            • *Takeaway:* Lower socioeconomic status strongly correlates with physical morbidity, chronic stress, and dietary barriers, producing a steep screening gradient.
            """
        )

with tab_mitigation:
    st.subheader("Fairness Mitigation Experiment (Sensitive Attribute: Income)")
    st.markdown(
        """
        To investigate algorithmic fairness mitigation, we conducted a post-processing experiment using 
        **Fairlearn's `ThresholdOptimizer`** with an **Equalized Odds** constraint on the **Income** attribute.
        
        **Methodological Protocol:**
        1. Underlying classifier: Pre-trained Logistic Regression baseline (`prefit=True`).
        2. Threshold optimization: Group-specific decision thresholds are solved on the **validation set only** (`X_val, y_val, Income_val`) to balance TPR and FPR across all 8 income brackets.
        3. Independent evaluation: Evaluated strictly on the **untouched test split** (`X_test, y_test, Income_test`).
        """
    )
    
    mit_info = meta.get("fairness_mitigation_experiment", {})
    if mit_info:
        col_m_chart, col_m_summary = st.columns([1.2, 1.0])
        with col_m_chart:
            st.plotly_chart(plot_mitigation_comparison(mit_info), use_container_width=True)
            
        with col_m_summary:
            st.markdown("##### Empirical Results Summary:")
            st.markdown(
                f"""
                | Metric | Before Mitigation | After Mitigation | Change |
                | :--- | :--- | :--- | :--- |
                | **Equalized Odds Diff** | **{mit_info['equalized_odds_diff_before']:.4f}** | **{mit_info['equalized_odds_diff_after']:.4f}** | **-78.1% (Massive Drop)** |
                | **Overall Accuracy** | **{mit_info['accuracy_before']:.2%}** | **{mit_info['accuracy_after']:.2%}** | **-5.95% (Accuracy Cost)** |
                """
            )
            st.markdown(
                """
                <div class="warning-callout">
                    <b>Operational Requirement at Inference Time:</b><br>
                    Post-processing mitigation requires that <b>patient income bracket is collected and supplied at runtime</b> to look up their specific decision threshold.<br><br>
                    <i>If a patient refuses to disclose their income, the mitigated model cannot produce a prediction without falling back to a default threshold.</i>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    st.markdown("---")
    st.subheader("The Fundamental Fairness-Utility Trade-Off")
    st.markdown(
        """
        Fairness mitigation is never a 'free lunch'. By forcing equal True Positive Rates and False Positive Rates 
        across income tiers:
        - The decision threshold for affluent cohorts was lowered (flagging more potential cases to match sensitivity).
        - The decision threshold for low-income cohorts was raised (suppressing false alarms).
        - **Net Result:** Equalized odds improved by 78%, but overall screening accuracy dropped from **74.76% to 68.81%**.
        
        *Healthcare Policy Implication:* Decision-makers must explicitly decide whether the goal is population-wide diagnostic accuracy or demographic error-rate parity.
        """
    )

with tab_concepts:
    st.subheader("Mathematical Definitions of Fairness Metrics")
    st.markdown(
        """
        #### 1. Demographic Parity (Statistical Parity)
        Demographic parity requires that the probability of being screened positive is independent of group membership $A$:
        $$P(\\hat{Y}=1 \\mid A=a) = P(\\hat{Y}=1 \\mid A=b) \\quad \\forall a, b$$
        $$\\text{Demographic Parity Difference} = \\max_a P(\\hat{Y}=1 \\mid A=a) - \\min_a P(\\hat{Y}=1 \\mid A=a)$$
        
        *Clinical Note:* In medicine, demographic parity is rarely appropriate when true biological prevalence varies substantially across groups (such as age). Enforcing equal positive screening rates would lead to severe under-screening of older adults and over-screening of teens.

        #### 2. Equalized Odds
        Equalized odds requires that predictor error rates are independent of group membership given the true disease state $Y$:
        $$P(\\hat{Y}=1 \\mid A=a, Y=y) = P(\\hat{Y}=1 \\mid A=b, Y=y) \\quad \\forall y \\in \\{0, 1\\}, \\; \\forall a, b$$
        $$\\text{Equalized Odds Difference} = \\max \\left( \\max_{a,b} |\\text{TPR}_a - \\text{TPR}_b|, \\; \\max_{a,b} |\\text{FPR}_a - \\text{FPR}_b| \\right)$$
        
        *Clinical Note:* Equalized odds ensures that a sick patient has an equal chance of being identified regardless of group, and a healthy patient has an equal chance of avoiding unnecessary follow-up procedures.
        """
    )
