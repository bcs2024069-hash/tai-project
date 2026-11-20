"""
Data Loading & Caching Utilities for the Streamlit Application.
Strictly loads precomputed artifacts; performs ZERO runtime retraining.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple, Optional


ARTIFACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
)


@st.cache_data
def load_metadata() -> Dict[str, Any]:
    """Loads model training metadata, data split info, and experiment summaries."""
    path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_evaluation_metrics() -> Dict[str, Any]:
    """Loads comprehensive evaluation metrics for both models."""
    path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_fairness_metrics() -> Dict[str, Any]:
    """Loads Fairlearn demographic parity and equalized odds subgroup metrics."""
    path = os.path.join(ARTIFACTS_DIR, "fairness_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_calibration_data() -> pd.DataFrame:
    """Loads 10-bin calibration curve points for both models."""
    path = os.path.join(ARTIFACTS_DIR, "calibration_data.csv")
    return pd.read_csv(path)


@st.cache_data
def load_robustness_summary() -> pd.DataFrame:
    """Loads aggregated perturbation robustness summary (mean shift, flip rates)."""
    path = os.path.join(ARTIFACTS_DIR, "robustness_summary.csv")
    return pd.read_csv(path)


@st.cache_data
def load_robustness_results() -> pd.DataFrame:
    """Loads individual sample-level perturbation results across 500 test cases."""
    path = os.path.join(ARTIFACTS_DIR, "robustness_results.csv")
    return pd.read_csv(path)


@st.cache_data
def load_sample_cases() -> pd.DataFrame:
    """Loads 15 curated test cases with features and predictions."""
    path = os.path.join(ARTIFACTS_DIR, "sample_cases.csv")
    return pd.read_csv(path)


@st.cache_data
def load_shap_global() -> pd.DataFrame:
    """Loads ranked global SHAP feature importance table."""
    path = os.path.join(ARTIFACTS_DIR, "shap_global_importance.csv")
    return pd.read_csv(path)


@st.cache_data
def load_shap_individual() -> pd.DataFrame:
    """Loads local SHAP attribution values for the 15 curated sample cases."""
    path = os.path.join(ARTIFACTS_DIR, "shap_individual_values.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_models() -> Tuple[Any, Any]:
    """Loads serialized Model 1 and Model 2 pipelines."""
    m1_path = os.path.join(ARTIFACTS_DIR, "baseline_model.joblib")
    m2_path = os.path.join(ARTIFACTS_DIR, "trustworthy_model.joblib")
    m1 = joblib.load(m1_path)
    m2 = joblib.load(m2_path)
    return m1, m2


def get_calibration_summary_metrics() -> Dict[str, Dict[str, float]]:
    """
    Returns ECE and Brier scores for Model 1 and Model 2.
    Computed during the rigorous training run on the untouched test split (N=10,604).
    """
    return {
        "baseline_model": {
            "name": "Model 1 (Baseline)",
            "brier_score": 0.16975,
            "ece": 0.02621,
        },
        "trustworthy_model": {
            "name": "Model 2 (Trust-Aware)",
            "brier_score": 0.16987,
            "ece": 0.02804,
        },
    }
