"""
Page 3: Individual Patient Case Predictions & Diagnostic Profile Explorer.
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
import plotly.graph_objects as go
from utils.helpers import (
    apply_custom_css,
    FEATURE_METADATA,
    format_feature_value,
)
from utils.data_loader import load_sample_cases

st.set_page_config(page_title="Case Predictions - Trustworthy Diabetes", layout="wide")
apply_custom_css()

st.title("Individual Patient Case Predictions & Clinical Profiles")
st.caption("Detailed Inspection of 15 Stratified Test Cases Across Both Screening Pipelines")

df_cases = load_sample_cases()

# Filter/Selector Bar
st.markdown("---")
c_sel1, c_sel2 = st.columns([1, 2])

with c_sel1:
    case_ids = df_cases["case_id"].tolist()
    selected_case_id = st.selectbox("Select Patient Case ID to Inspect:", case_ids)

selected_row = df_cases[df_cases["case_id"] == selected_case_id].iloc[0]

with c_sel2:
    actual = int(selected_row["actual_label"])
    m1_pred = int(selected_row["model1_pred"])
    m2_pred = int(selected_row["model2_pred"])
    
    actual_str = "Diabetic / Pre-diabetic (1)" if actual == 1 else "Non-Diabetic (0)"
    m1_str = "Screen Positive (1)" if m1_pred == 1 else "Screen Negative (0)"
    m2_str = "Screen Positive (1)" if m2_pred == 1 else "Screen Negative (0)"
    
    m1_badge = "Correct" if selected_row["model1_correct"] else "Misclassified"
    m2_badge = "Correct" if selected_row["model2_correct"] else "Misclassified"
    
    st.markdown(
        f"""
        **Case Summary:**  
        • **Ground Truth Outcome:** <span class="badge-neutral">{actual_str}</span>  
        • **Model 1 Decision:** {m1_str} — {m1_badge}  
        • **Model 2 Decision:** {m2_str} — {m2_badge}
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Risk Probability Comparison Gauges
col_g1, col_g2 = st.columns(2)

def create_gauge(prob: float, title: str, pred: int, actual: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>Threshold: 50%</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#dc2626" if prob >= 0.5 else "#1e40af"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 50], "color": "#fef9c3"},
                    {"range": [50, 70], "color": "#ffedd5"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=20))
    return fig

with col_g1:
    st.plotly_chart(
        create_gauge(
            selected_row["model1_prob"],
            "Model 1 (Baseline LR Risk)",
            m1_pred,
            actual,
        ),
        use_container_width=True,
    )

with col_g2:
    st.plotly_chart(
        create_gauge(
            selected_row["model2_prob"],
            "Model 2 (Calibrated LR Risk)",
            m2_pred,
            actual,
        ),
        use_container_width=True,
    )

# Patient Clinical & Behavioral Profile
st.subheader("Patient Clinical & Demographic Profile")

profile_c1, profile_c2, profile_c3 = st.columns(3)

with profile_c1:
    st.markdown("##### Physiological & Clinical Markers")
    st.markdown(f"• **High Blood Pressure:** {format_feature_value('HighBP', selected_row['HighBP'])}")
    st.markdown(f"• **High Cholesterol:** {format_feature_value('HighChol', selected_row['HighChol'])}")
    st.markdown(f"• **Cholesterol Checked (5 yrs):** {format_feature_value('CholCheck', selected_row['CholCheck'])}")
    st.markdown(f"• **Body Mass Index (BMI):** {selected_row['BMI']:.1f} kg/m²")
    st.markdown(f"• **Difficulty Walking / Stairs:** {format_feature_value('DiffWalk', selected_row['DiffWalk'])}")

with profile_c2:
    st.markdown("##### Lifestyle & Medical History")
    st.markdown(f"• **Smoked ≥ 100 Cigarettes:** {format_feature_value('Smoker', selected_row['Smoker'])}")
    st.markdown(f"• **Physical Exercise (30 Days):** {format_feature_value('PhysActivity', selected_row['PhysActivity'])}")
    st.markdown(f"• **Daily Fruit Intake:** {format_feature_value('Fruits', selected_row['Fruits'])}")
    st.markdown(f"• **Daily Veggie Intake:** {format_feature_value('Veggies', selected_row['Veggies'])}")
    st.markdown(f"• **Heavy Alcohol Consumption:** {format_feature_value('HvyAlcoholConsump', selected_row['HvyAlcoholConsump'])}")
    st.markdown(f"• **History of Stroke:** {format_feature_value('Stroke', selected_row['Stroke'])}")
    st.markdown(f"• **Coronary Heart Disease / MI:** {format_feature_value('HeartDiseaseorAttack', selected_row['HeartDiseaseorAttack'])}")

with profile_c3:
    st.markdown("##### Self-Reported Health & Demographics")
    st.markdown(f"• **General Health Rating:** {format_feature_value('GenHlth', selected_row['GenHlth'])}")
    st.markdown(f"• **Physical Illness Days (Past 30):** {int(selected_row['PhysHlth'])} days")
    st.markdown(f"• **Mental Stress Days (Past 30):** {int(selected_row['MentHlth'])} days")
    st.markdown(f"• **Biological Sex:** {format_feature_value('Sex', selected_row['Sex'])}")
    st.markdown(f"• **Age Bracket:** {format_feature_value('Age', selected_row['Age'])}")
    st.markdown(f"• **Education Level:** {format_feature_value('Education', selected_row['Education'])}")
    st.markdown(f"• **Household Income:** {format_feature_value('Income', selected_row['Income'])}")
    st.markdown(f"• **Healthcare Coverage:** {format_feature_value('AnyHealthcare', selected_row['AnyHealthcare'])}")
    st.markdown(f"• **Doctor Cost Barrier:** {format_feature_value('NoDocbcCost', selected_row['NoDocbcCost'])}")

st.markdown("---")

# Full Cohort Sample Table
st.subheader("All 15 Curated Test Cases")
st.write("Browse the stratified test cases including true positives, true negatives, and borderline classification cases:")

filter_option = st.radio(
    "Filter Cases:",
    ["All 15 Cases", "Both Models Correct", "Disagreements or Misclassifications"],
    horizontal=True,
)

display_df = df_cases.copy()
if filter_option == "Both Models Correct":
    display_df = display_df[display_df["model1_correct"] & display_df["model2_correct"]]
elif filter_option == "Disagreements or Misclassifications":
    display_df = display_df[(~display_df["model1_correct"]) | (~display_df["model2_correct"]) | (display_df["model1_pred"] != display_df["model2_pred"])]

cols_to_show = [
    "case_id", "actual_label", "model1_prob", "model1_pred", "model1_correct",
    "model2_prob", "model2_pred", "model2_correct", "BMI", "GenHlth", "Age", "HighBP", "HighChol"
]
st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)
