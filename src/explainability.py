"""
Explainability Module for Trustworthy Diabetes Risk-Screening.

This module computes SHAP (SHapley Additive exPlanations) values for the
Trust-Aware model.

IMPORTANT ARCHITECTURAL NOTE:
The primary deployed model is calibrated via CalibratedClassifierCV(method='sigmoid').
Because CalibratedClassifierCV applies a monotonic sigmoid transformation to the
inner linear model's logits, computing exact SHAP directly on the calibrated probability
space is non-linear and computationally intractable without sampling approximations.
Therefore, this module extracts the inner pipeline's fitted LogisticRegression classifier
and explains its uncalibrated log-odds score using shap.LinearExplainer.
The resulting SHAP values reflect the contribution of each feature to the pre-calibration
decision score (log-odds), preserving feature attribution fidelity and interpretability.
"""

import os
from typing import List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import shap


def _extract_inner_pipeline(calibrated_model):
    """Safely extracts the inner Pipeline from CalibratedClassifierCV or FrozenEstimator."""
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        inner = calibrated_model.calibrated_classifiers_[0].estimator
        if hasattr(inner, "estimator"):
            inner = inner.estimator
        return inner
    elif hasattr(calibrated_model, "named_steps"):
        return calibrated_model
    elif hasattr(calibrated_model, "estimator"):
        inner = calibrated_model.estimator
        if hasattr(inner, "named_steps"):
            return inner
        return _extract_inner_pipeline(inner)
    else:
        raise ValueError("Model format not recognized: expected CalibratedClassifierCV or Pipeline.")


def compute_global_shap(
    calibrated_model,
    X_train_sample: Union[pd.DataFrame, np.ndarray],
    feature_names_transformed: List[str],
    n_background: int = 200,
    n_explain: int = 500,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes global SHAP values using the inner logistic regression classifier.

    Extracts the inner pipeline from the calibrated classifier, applies the fitted
    preprocessor to background and explanation samples, and runs shap.LinearExplainer.

    NOTE: Explains the uncalibrated log-odds score of the inner model,
    not the calibrated probability.

    Parameters
    ----------
    calibrated_model : CalibratedClassifierCV or Pipeline
        The trained calibrated model or an inner pipeline.
    X_train_sample : pd.DataFrame or np.ndarray
        Sample of raw input training data used for background reference and explanation.
    feature_names_transformed : List[str]
        Ordered feature names matching the transformed feature space.
    n_background : int, default=200
        Number of background reference samples.
    n_explain : int, default=500
        Number of samples to explain.
    random_state : int, default=42
        Random seed for sampling.

    Returns
    -------
    shap_values_pos_class : np.ndarray
        2D array of shape (n_explain, n_features) of SHAP values for the positive class (Diabetes).
    feature_names_transformed : List[str]
        The transformed feature names corresponding to columns of shap_values.
    """
    inner_pipeline = _extract_inner_pipeline(calibrated_model)
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    classifier = inner_pipeline.named_steps["classifier"]

    total_rows = len(X_train_sample)
    rng = np.random.RandomState(random_state)
    bg_n = min(n_background, total_rows)
    exp_n = min(n_explain, total_rows)

    if isinstance(X_train_sample, pd.DataFrame):
        bg_idx = rng.choice(total_rows, size=bg_n, replace=False)
        exp_idx = rng.choice(total_rows, size=exp_n, replace=False)
        X_bg = X_train_sample.iloc[bg_idx]
        X_exp = X_train_sample.iloc[exp_idx]
    else:
        bg_idx = rng.choice(total_rows, size=bg_n, replace=False)
        exp_idx = rng.choice(total_rows, size=exp_n, replace=False)
        X_bg = X_train_sample[bg_idx]
        X_exp = X_train_sample[exp_idx]

    X_bg_transformed = preprocessor.transform(X_bg)
    X_exp_transformed = preprocessor.transform(X_exp)

    explainer = shap.LinearExplainer(classifier, X_bg_transformed)
    shap_vals = explainer.shap_values(X_exp_transformed)

    if isinstance(shap_vals, list):
        shap_values_pos_class = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_values_pos_class = shap_vals[:, :, 1]
    else:
        shap_values_pos_class = shap_vals

    return shap_values_pos_class, feature_names_transformed


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes global feature importance as mean absolute SHAP value per feature.

    Parameters
    ----------
    shap_values : np.ndarray
        Array of shape (n_samples, n_features).
    feature_names : List[str]
        List of feature names matching the columns of shap_values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'mean_abs_shap', 'rank'],
        sorted descending by mean_abs_shap.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })
    df = df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["feature", "mean_abs_shap", "rank"]]


def compute_local_shap(
    calibrated_model,
    X_records: Union[pd.DataFrame, np.ndarray],
    feature_names_transformed: List[str],
    X_train_background: Union[pd.DataFrame, np.ndarray],
    n_background: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Computes local SHAP attributions for selected individual patient records.

    Parameters
    ----------
    calibrated_model : CalibratedClassifierCV or Pipeline
        The model containing the fitted pipeline.
    X_records : pd.DataFrame or np.ndarray
        The individual records to explain.
    feature_names_transformed : List[str]
        Ordered feature names corresponding to the preprocessor output.
    X_train_background : pd.DataFrame or np.ndarray
        Reference training data for background expectations.
    n_background : int, default=200
        Number of background reference rows.
    random_state : int, default=42
        Random seed for background sampling.

    Returns
    -------
    shap_values : np.ndarray
        Array of shape (n_records, n_features) of SHAP values.
    base_value : float
        The background baseline expectation value (expected log-odds).
    """
    inner_pipeline = _extract_inner_pipeline(calibrated_model)
    preprocessor = inner_pipeline.named_steps["preprocessor"]
    classifier = inner_pipeline.named_steps["classifier"]

    total_rows = len(X_train_background)
    bg_n = min(n_background, total_rows)
    rng = np.random.RandomState(random_state)

    if isinstance(X_train_background, pd.DataFrame):
        bg_idx = rng.choice(total_rows, size=bg_n, replace=False)
        X_bg = X_train_background.iloc[bg_idx]
    else:
        bg_idx = rng.choice(total_rows, size=bg_n, replace=False)
        X_bg = X_train_background[bg_idx]

    X_bg_transformed = preprocessor.transform(X_bg)
    X_rec_transformed = preprocessor.transform(X_records)

    explainer = shap.LinearExplainer(classifier, X_bg_transformed)
    shap_vals = explainer.shap_values(X_rec_transformed)

    if isinstance(shap_vals, list):
        shap_values_pos = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_values_pos = shap_vals[:, :, 1]
    else:
        shap_values_pos = shap_vals

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_value = float(ev[1] if len(ev) > 1 else ev[0])
    else:
        base_value = float(ev)

    return shap_values_pos, base_value


def save_shap_artifacts(
    global_importance_df: pd.DataFrame,
    local_shap_values: np.ndarray,
    feature_names: List[str],
    case_ids: List[Union[str, int]],
    artifacts_dir: str = "artifacts"
) -> Tuple[str, str]:
    """
    Saves SHAP global importance and local attribution artifacts to CSV.

    Parameters
    ----------
    global_importance_df : pd.DataFrame
        DataFrame with ['feature', 'mean_abs_shap', 'rank'].
    local_shap_values : np.ndarray
        Array of shape (len(case_ids), len(feature_names)).
    feature_names : List[str]
        List of feature names matching columns.
    case_ids : List[Union[str, int]]
        Identifiers for each local record explained.
    artifacts_dir : str, default='artifacts'
        Directory to save artifacts in.

    Returns
    -------
    global_path : str
        Path to shap_global_importance.csv.
    individual_path : str
        Path to shap_individual_values.csv.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    global_path = os.path.join(artifacts_dir, "shap_global_importance.csv")
    global_importance_df.to_csv(global_path, index=False)

    individual_df = pd.DataFrame(local_shap_values, columns=feature_names)
    individual_df.insert(0, "case_id", case_ids)
    individual_path = os.path.join(artifacts_dir, "shap_individual_values.csv")
    individual_df.to_csv(individual_path, index=False)

    return global_path, individual_path
