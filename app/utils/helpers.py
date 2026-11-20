"""
Helper utilities, UI components, codebook definitions, and clinical disclaimers
for the Trustworthy Diabetes Risk-Screening Decision System.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional


# -----------------------------------------------------------------------------
# Clinical Disclaimer & Educational Warning (Disabled)
# -----------------------------------------------------------------------------
DISCLAIMER_TEXT = ""

def render_clinical_disclaimer(expanded: bool = False):
    """No-op: Medical disclaimer has been removed from all dashboard pages."""
    pass


# -----------------------------------------------------------------------------
# BRFSS Codebook & Data Dictionary
# -----------------------------------------------------------------------------
FEATURE_METADATA = {
    "HighBP": {
        "name": "High Blood Pressure",
        "category": "Clinical Indicator",
        "type": "Binary",
        "description": "Have you ever been told by a doctor, nurse, or other health professional that you have high blood pressure?",
        "values": {0: "No", 1: "Yes"},
    },
    "HighChol": {
        "name": "High Cholesterol",
        "category": "Clinical Indicator",
        "type": "Binary",
        "description": "Have you ever been told by a doctor or health professional that your blood cholesterol is high?",
        "values": {0: "No", 1: "Yes"},
    },
    "CholCheck": {
        "name": "Cholesterol Check (5 Yrs)",
        "category": "Healthcare Access",
        "type": "Binary",
        "description": "Have you had your cholesterol checked within the past 5 years?",
        "values": {0: "No", 1: "Yes"},
    },
    "BMI": {
        "name": "Body Mass Index (BMI)",
        "category": "Physical Measurement",
        "type": "Continuous",
        "description": "Body Mass Index calculated as weight in kilograms divided by height in meters squared (kg/m²).",
        "range": "12 to 98 (typical: 18.5 - 40+)",
    },
    "Smoker": {
        "name": "Smoked >= 100 Cigarettes",
        "category": "Behavioral Risk",
        "type": "Binary",
        "description": "Have you smoked at least 100 cigarettes (5 packs) in your entire life?",
        "values": {0: "No", 1: "Yes"},
    },
    "Stroke": {
        "name": "History of Stroke",
        "category": "Cardiovascular History",
        "type": "Binary",
        "description": "Have you ever been told that you had a stroke?",
        "values": {0: "No", 1: "Yes"},
    },
    "HeartDiseaseorAttack": {
        "name": "Coronary Heart Disease / MI",
        "category": "Cardiovascular History",
        "type": "Binary",
        "description": "Ever reported coronary heart disease (CHD) or myocardial infarction (heart attack)?",
        "values": {0: "No", 1: "Yes"},
    },
    "PhysActivity": {
        "name": "Physical Activity (30 Days)",
        "category": "Behavioral Risk",
        "type": "Binary",
        "description": "During the past 30 days, did you participate in any physical activities or exercise outside your job?",
        "values": {0: "No", 1: "Yes"},
    },
    "Fruits": {
        "name": "Fruit Consumption",
        "category": "Dietary Habit",
        "type": "Binary",
        "description": "Consume fruit 1 or more times per day?",
        "values": {0: "No", 1: "Yes"},
    },
    "Veggies": {
        "name": "Vegetable Consumption",
        "category": "Dietary Habit",
        "type": "Binary",
        "description": "Consume vegetables 1 or more times per day?",
        "values": {0: "No", 1: "Yes"},
    },
    "HvyAlcoholConsump": {
        "name": "Heavy Alcohol Consumption",
        "category": "Behavioral Risk",
        "type": "Binary",
        "description": "Adult men >14 drinks/week; adult women >7 drinks/week.",
        "values": {0: "No", 1: "Yes"},
    },
    "AnyHealthcare": {
        "name": "Healthcare Coverage",
        "category": "Healthcare Access",
        "type": "Binary",
        "description": "Do you have any kind of health care coverage (health insurance, prepaid plans, Medicare)?",
        "values": {0: "No", 1: "Yes"},
    },
    "NoDocbcCost": {
        "name": "Doctor Visit Cost Barrier",
        "category": "Healthcare Access",
        "type": "Binary",
        "description": "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
        "values": {0: "No", 1: "Yes"},
    },
    "GenHlth": {
        "name": "General Health Rating",
        "category": "Self-Reported Health",
        "type": "Ordinal (1-5)",
        "description": "Would you say that in general your health is: Excellent (1), Very Good (2), Good (3), Fair (4), Poor (5)?",
        "values": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
    },
    "MentHlth": {
        "name": "Poor Mental Health Days",
        "category": "Self-Reported Health",
        "type": "Discrete (0-30)",
        "description": "Now thinking about your mental health (stress, depression, problems with emotions), how many days during the past 30 days was your mental health not good?",
        "range": "0 to 30 days",
    },
    "PhysHlth": {
        "name": "Poor Physical Health Days",
        "category": "Self-Reported Health",
        "type": "Discrete (0-30)",
        "description": "Now thinking about your physical health (physical illness and injury), how many days during the past 30 days was your physical health not good?",
        "range": "0 to 30 days",
    },
    "DiffWalk": {
        "name": "Difficulty Walking / Stairs",
        "category": "Physical Measurement",
        "type": "Binary",
        "description": "Do you have serious difficulty walking or climbing stairs?",
        "values": {0: "No", 1: "Yes"},
    },
    "Sex": {
        "name": "Biological Sex",
        "category": "Demographic",
        "type": "Binary",
        "description": "Biological sex reported by respondent.",
        "values": {0: "Female", 1: "Male"},
    },
    "Age": {
        "name": "Age Category",
        "category": "Demographic",
        "type": "Ordinal (1-13)",
        "description": "13-level age classification starting from 18-24 up to 80+.",
        "values": {
            1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
            6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
            11: "70-74", 12: "75-79", 13: "80+"
        },
    },
    "Education": {
        "name": "Education Level",
        "category": "Socioeconomic",
        "type": "Ordinal (1-6)",
        "description": "Highest level of school completed.",
        "values": {
            1: "Never / Kindergarten",
            2: "Elementary (Grades 1-8)",
            3: "Some High School (Grades 9-11)",
            4: "High School Graduate",
            5: "Some College / Tech School",
            6: "College Graduate (4+ yrs)"
        },
    },
    "Income": {
        "name": "Annual Household Income",
        "category": "Socioeconomic",
        "type": "Ordinal (1-8)",
        "description": "Annual household income bracket from all sources.",
        "values": {
            1: "<$10,000",
            2: "$10,000 - $14,999",
            3: "$15,000 - $19,999",
            4: "$20,000 - $24,999",
            5: "$25,000 - $34,999",
            6: "$35,000 - $49,999",
            7: "$50,000 - $74,999",
            8: "≥$75,000"
        },
    },
}

def format_feature_value(feature: str, val: Any) -> str:
    """Formats raw feature values into human-readable survey strings."""
    if feature not in FEATURE_METADATA:
        return str(val)
    meta = FEATURE_METADATA[feature]
    if "values" in meta:
        try:
            int_val = int(val)
            return meta["values"].get(int_val, str(val))
        except (ValueError, TypeError):
            return str(val)
    if feature in ["PhysHlth", "MentHlth"]:
        return f"{int(val)} days"
    if feature == "BMI":
        return f"{float(val):.1f} kg/m²"
    return str(val)


# -----------------------------------------------------------------------------
# Common UI Styling
# -----------------------------------------------------------------------------
def apply_custom_css():
    """Injects refined CSS styling for clean, readable medical research dashboards."""
    st.markdown(
        """
        <style>
        /* Card metrics */
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetric"]:hover {
            border-color: #cbd5e1;
        }
        
        /* Badges */
        .badge-positive {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-negative {
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-neutral {
            background-color: #f1f5f9;
            color: #475569;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        /* Callout boxes */
        .info-callout {
            background-color: #f0fdf4;
            border-left: 4px solid #16a34a;
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
            margin-bottom: 1rem;
        }
        .warning-callout {
            background-color: #fffbeb;
            border-left: 4px solid #d97706;
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
            margin-bottom: 1rem;
        }
        .disclaimer-banner {
            background-color: #fff1f2;
            border: 1px solid #fecdd3;
            border-left: 5px solid #e11d48;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
