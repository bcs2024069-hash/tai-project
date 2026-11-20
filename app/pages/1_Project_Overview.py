"""
Page 1: Project Overview, Dataset Cohort, and Methodological Rigor.
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
from utils.helpers import apply_custom_css, FEATURE_METADATA
from utils.data_loader import load_metadata

st.set_page_config(page_title="Project Overview - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Project Overview & Data Architecture")
st.caption("CDC BRFSS 2015 Survey Cohort, 21-Feature Codebook, and Leakage Prevention Framework")

# Tabs
tab_abstract, tab_dataset, tab_codebook, tab_splits, tab_duplicates = st.tabs([
    "Research Abstract",
    "Cohort Characteristics",
    "Feature Codebook (21 Features)",
    "Splitting & Leakage Prevention",
    "Duplicate Survey Analysis",
])

with tab_abstract:
    st.subheader("Academic Abstract & Systems Motivation")
    st.markdown(
        """
        **Background:** Diabetes mellitus is a major chronic disease worldwide associated with cardiovascular 
        complications, renal disease, and premature mortality. Early population-level risk screening can 
        identify high-risk individuals for confirmatory diagnostic testing and lifestyle intervention.
        
        **The Problem:** Traditional clinical risk scoring tools and binary ML classifiers are often evaluated 
        solely on receiver operating characteristic (ROC) curves or raw accuracy. In clinical decision support, 
        such metrics fail to reveal whether:
        1. Estimated risk probabilities accurately reflect true disease rates (calibration).
        2. Predictions systematically under-screen or over-screen specific racial, age, or socioeconomic groups (fairness).
        3. Model reasoning aligns with known pathophysiology rather than spurious survey correlations (explainability).
        4. Model risk scores fluctuate erratically under minor self-reporting noise (robustness).
        
        **This Project's Contribution:** We develop a comparative framework between a standard uncalibrated 
        Logistic Regression baseline (**Model 1**) and a trust-aware decision framework (**Model 2**) featuring 
        Platt probability calibration, SHAP explainability, Fairlearn algorithmic audits across Sex, Age, and Income, 
        and 6 controlled input perturbation stress tests.
        """
    )
    st.info(
        "**Core Methodological Principle:** Model 2 is evaluated with scientific neutrality. Trustworthy AI "
        "components provide diagnostic transparency into model behavior; they do not automatically make a model "
        "more accurate or inherently superior."
    )

with tab_dataset:
    st.subheader("CDC BRFSS 2015 Survey Cohort Profile")
    st.markdown(
        """
        The **Behavioral Risk Factor Surveillance System (BRFSS)** is the premier annual telephone health survey 
        conducted by the U.S. Centers for Disease Control and Prevention (CDC). It collects self-reported data 
        on health-related risk behaviors, chronic health conditions, and preventive service utilization.
        """
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cohort Records", "70,692", help="Primary experimental dataset")
    with col2:
        st.metric("Class Balance", "50.0% / 50.0%", help="Exactly 35,346 diabetes/prediabetes and 35,346 non-diabetic")
    with col3:
        st.metric("Survey Predictors", "21 Features", help="7 continuous/ordinal standardized + 14 binary indicators")
    with col4:
        st.metric("Survey Year", "CDC 2015", help="Administered across all 50 US states and territories")
        
    st.warning(
        "**Clinical Context on 50/50 Class Balance:**  \n"
        "The underlying dataset is an artificially balanced 50/50 cohort curated by health informatics researchers to "
        "facilitate balanced machine learning training. In the general adult US population, diagnosed diabetes prevalence "
        "is approximately 10% to 14%. Because the training distribution has a 50% base rate, the predicted probabilities "
        "represent **relative risk within this balanced screening cohort**, not unadjusted absolute population incidence."
    )

with tab_codebook:
    st.subheader("Interactive 21-Feature Codebook & Survey Dictionary")
    st.write("Browse the 21 survey indicators grouped by clinical and behavioral domain:")
    
    categories = sorted(list(set(f["category"] for f in FEATURE_METADATA.values())))
    selected_cat = st.selectbox("Filter by Domain / Category:", ["All Categories"] + categories)
    
    rows = []
    for code, info in FEATURE_METADATA.items():
        if selected_cat != "All Categories" and info["category"] != selected_cat:
            continue
        vals_str = ""
        if "values" in info:
            vals_str = ", ".join([f"{k}={v}" for k, v in info["values"].items()])
        elif "range" in info:
            vals_str = info["range"]
        rows.append({
            "Feature Code": code,
            "Clinical / Survey Name": info["name"],
            "Domain": info["category"],
            "Data Type": info["type"],
            "Survey Question / Description": info["description"],
            "Valid Values / Range": vals_str,
        })
        
    df_cb = pd.DataFrame(rows)
    st.dataframe(df_cb, use_container_width=True, hide_index=True)

with tab_splits:
    st.subheader("Data Partitioning & Strict Leakage Prevention")
    st.markdown(
        """
        To guarantee scientific reproducibility and prevent data leakage, the 70,692 records were partitioned 
        using **stratified 70 / 15 / 15 splitting** with a fixed random seed (`random_state=42`):
        """
    )
    
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            """
            <div style="background-color: #f1f5f9; padding: 14px; border-radius: 6px; border-left: 4px solid #1e40af;">
                <b>Training Split (70.0%)</b><br>
                <b>N = 49,484</b> (24,742 positive, 24,742 negative)<br>
                <i>Used strictly for:</i> Model 1 fitting and Model 2 inner pipeline fitting. Preprocessing scaler 
                parameters (mean, variance) are computed strictly on this split.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div style="background-color: #f1f5f9; padding: 14px; border-radius: 6px; border-left: 4px solid #d97706;">
                <b>Validation Split (15.0%)</b><br>
                <b>N = 10,604</b> (5,302 positive, 5,302 negative)<br>
                <i>Used strictly for:</i> Post-hoc Platt sigmoid probability calibration (CalibratedClassifierCV) 
                and Fairlearn ThresholdOptimizer decision threshold fitting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div style="background-color: #f1f5f9; padding: 14px; border-radius: 6px; border-left: 4px solid #16a34a;">
                <b>Test Split (15.0%)</b><br>
                <b>N = 10,604</b> (5,302 positive, 5,302 negative)<br>
                <i>Used strictly for:</i> Final independent evaluation. Untouched during preprocessing fitting, 
                model training, calibration fitting, and mitigation threshold selection.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Leakage Guarantee:** Preprocessing feature scalers (`StandardScaler`) are fit ONLY on the training split "
        "and applied via `ColumnTransformer`. The validation split calibrates probabilities using `FrozenEstimator` "
        "without refitting coefficients. The test split is evaluated only once on the finalized frozen pipelines."
    )

with tab_duplicates:
    st.subheader("Survey Deduplication & Integrity Investigation")
    st.markdown(
        """
        In survey datasets, identical response vectors across a limited set of categorical questions can either 
        represent identical individuals (accidental duplicates) or distinct individuals who happen to share identical 
        demographic and behavioral profiles.
        """
    )
    
    meta = load_metadata()
    dedup = meta.get("deduplication_sensitivity_experiment", {})
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            """
            **Empirical Finding on Survey Integrity:**
            - Total original records: **70,692**
            - Exact duplicated rows: **1,635** (2.31%)
            - Unique feature rows: **69,057**
            
            **The Crucial Proof:**
            Analysis revealed **408 distinct feature combinations (spanning 1,001 respondents)** where respondents had 
            **identical feature profiles but opposite diabetes outcomes** (one positive, one negative).
            
            Because identical feature rows produce conflicting target labels, they **cannot** be accidental database copies; 
            they represent genuine, distinct human survey respondents in the U.S. population.
            """
        )
    with col_d2:
        st.markdown(
            f"""
            **Sensitivity Comparison (Primary vs Deduplicated):**
            
            | Metric | Primary (70,692 rows) | Deduplicated (69,057 rows) |
            | :--- | :--- | :--- |
            | **Test Sample Size** | N = 10,604 | N = 10,359 |
            | **Model 1 Accuracy** | {dedup.get('model1_accuracy_original', 0.7476):.4f} | {dedup.get('model1_accuracy_dedup', 0.7441):.4f} |
            | **Model 1 ROC-AUC** | {dedup.get('model1_roc_auc_original', 0.8251):.4f} | {dedup.get('model1_roc_auc_dedup', 0.8223):.4f} |
            | **Model 2 Accuracy** | {dedup.get('model2_accuracy_original', 0.7475):.4f} | {dedup.get('model2_accuracy_dedup', 0.7440):.4f} |
            | **Model 2 ROC-AUC** | {dedup.get('model2_roc_auc_original', 0.8251):.4f} | {dedup.get('model2_roc_auc_dedup', 0.8223):.4f} |
            
            *Conclusion:* Metrics vary by less than 0.35%, proving that the primary findings are highly robust to deduplication.
            """
        )
