"""
Calibration Evaluation Module for Diabetes Risk-Screening.

This module computes calibration curves, Brier scores, and Expected Calibration Error (ECE)
to evaluate probability reliability for clinical decision support.

Metrics:
- Reliability diagram points: fraction of positives vs mean predicted value
- Brier Score Loss: mean squared error between probabilities and binary outcomes
- Expected Calibration Error (ECE): weighted average difference between accuracy and confidence across bins
- Predicted probability separation across ground-truth classes
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_curve(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    n_bins: int = 10,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes calibration curve points, Brier score, and Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary target values.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of positive class.
    n_bins : int, default=10
        Number of bins for discretization.
    model_name : str, default='Model'
        Label for the model.

    Returns
    -------
    dict
        Contains 'model_name', 'fraction_of_positives', 'mean_predicted_value',
        'brier_score', 'ece', and 'n_bins'.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    total_n = len(y_t)

    # 1. Scikit-learn calibration curve
    frac_pos, mean_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy="uniform")

    # 2. Brier score
    brier = float(brier_score_loss(y_t, y_p))

    # 3. Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_p, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = (bin_indices == b)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (n_b / total_n) * np.abs(bin_acc - bin_conf)

    return {
        "model_name": model_name,
        "fraction_of_positives": [round(float(x), 4) for x in frac_pos],
        "mean_predicted_value": [round(float(x), 4) for x in mean_pred],
        "brier_score": round(brier, 5),
        "ece": round(float(ece), 5),
        "n_bins": n_bins,
    }


def compute_probability_distribution(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes summary statistics and distributions of predicted probabilities
    separated by true negative (y=0) and true positive (y=1) outcomes.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted probabilities.
    model_name : str, default='Model'
        Model identifier.

    Returns
    -------
    dict
        Distribution statistics per class.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)

    prob_neg = y_p[y_t == 0]
    prob_pos = y_p[y_t == 1]

    return {
        "model_name": model_name,
        "n_negatives": int(len(prob_neg)),
        "n_positives": int(len(prob_pos)),
        "mean_prob_negative": round(float(np.mean(prob_neg)), 4) if len(prob_neg) > 0 else 0.0,
        "mean_prob_positive": round(float(np.mean(prob_pos)), 4) if len(prob_pos) > 0 else 0.0,
        "std_prob_negative": round(float(np.std(prob_neg)), 4) if len(prob_neg) > 0 else 0.0,
        "std_prob_positive": round(float(np.std(prob_pos)), 4) if len(prob_pos) > 0 else 0.0,
        "median_prob_negative": round(float(np.median(prob_neg)), 4) if len(prob_neg) > 0 else 0.0,
        "median_prob_positive": round(float(np.median(prob_pos)), 4) if len(prob_pos) > 0 else 0.0,
    }


def compute_calibration_both_models(
    y_true: Union[pd.Series, np.ndarray],
    y_prob_baseline: Union[pd.Series, np.ndarray],
    y_prob_trustworthy: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes calibration curves and distribution metrics for both Baseline and Trust-Aware models.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_prob_baseline : array-like
        Baseline model positive class probabilities.
    y_prob_trustworthy : array-like
        Trust-aware model positive class probabilities.
    n_bins : int, default=10
        Number of calibration bins.

    Returns
    -------
    dict
        Comparative calibration report.
    """
    cal_base = compute_calibration_curve(
        y_true, y_prob_baseline, n_bins=n_bins, model_name="Baseline (Model 1)"
    )
    cal_trust = compute_calibration_curve(
        y_true, y_prob_trustworthy, n_bins=n_bins, model_name="Trust-Aware (Model 2)"
    )

    dist_base = compute_probability_distribution(
        y_true, y_prob_baseline, model_name="Baseline (Model 1)"
    )
    dist_trust = compute_probability_distribution(
        y_true, y_prob_trustworthy, model_name="Trust-Aware (Model 2)"
    )

    summary = {
        "baseline_brier": cal_base["brier_score"],
        "trustworthy_brier": cal_trust["brier_score"],
        "brier_improvement": round(cal_base["brier_score"] - cal_trust["brier_score"], 5),
        "baseline_ece": cal_base["ece"],
        "trustworthy_ece": cal_trust["ece"],
        "ece_improvement": round(cal_base["ece"] - cal_trust["ece"], 5),
    }

    return {
        "baseline_calibration": cal_base,
        "trustworthy_calibration": cal_trust,
        "baseline_distribution": dist_base,
        "trustworthy_distribution": dist_trust,
        "summary": summary,
    }


def save_calibration_artifacts(
    calibration_dict: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> str:
    """
    Saves calibration comparison data points to artifacts/calibration_data.csv.

    Parameters
    ----------
    calibration_dict : dict
        Dictionary returned by compute_calibration_both_models or compute_calibration_curve.
    artifacts_dir : str, default='artifacts'
        Target directory.

    Returns
    -------
    str
        Path to saved CSV artifact.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "calibration_data.csv")

    # If comparison dictionary is passed
    if "baseline_calibration" in calibration_dict and "trustworthy_calibration" in calibration_dict:
        b_cal = calibration_dict["baseline_calibration"]
        t_cal = calibration_dict["trustworthy_calibration"]

        max_len = max(len(b_cal["mean_predicted_value"]), len(t_cal["mean_predicted_value"]))

        rows = []
        for i in range(max_len):
            row = {"bin_index": i + 1}
            if i < len(b_cal["mean_predicted_value"]):
                row["baseline_mean_pred"] = b_cal["mean_predicted_value"][i]
                row["baseline_frac_pos"] = b_cal["fraction_of_positives"][i]
            else:
                row["baseline_mean_pred"] = None
                row["baseline_frac_pos"] = None

            if i < len(t_cal["mean_predicted_value"]):
                row["trustworthy_mean_pred"] = t_cal["mean_predicted_value"][i]
                row["trustworthy_frac_pos"] = t_cal["fraction_of_positives"][i]
            else:
                row["trustworthy_mean_pred"] = None
                row["trustworthy_frac_pos"] = None

            rows.append(row)

        df = pd.DataFrame(rows)
    else:
        # Single model passed
        frac_pos = calibration_dict.get("fraction_of_positives", [])
        mean_pred = calibration_dict.get("mean_predicted_value", [])
        df = pd.DataFrame({
            "bin_index": range(1, len(mean_pred) + 1),
            "mean_predicted_value": mean_pred,
            "fraction_of_positives": frac_pos,
        })

    df.to_csv(out_path, index=False)
    return out_path
