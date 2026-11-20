"""
Fairness and Subgroup Disparity Evaluation Module.

Evaluates algorithmic fairness across protected demographic attributes:
- Sex (0: Female, 1: Male)
- Age (1: 18-24 through 13: 80+)
- Income (1: <$10k through 8: $75k+)

Note: The CDC BRFSS dataset does not contain race/ethnicity features.
Fairness criteria computed:
- Demographic Parity Difference: max(selection_rate) - min(selection_rate)
- Equalized Odds Difference: max(max(TPR) - min(TPR), max(FPR) - min(FPR))
- True Positive Rate (Sensitivity), False Positive Rate, False Negative Rate, Selection Rate
"""

import json
import os
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd


SEX_LABELS = {0: "Female", 1: "Male"}

AGE_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

INCOME_LABELS = {
    1: "<$10k",
    2: "$10-15k",
    3: "$15-20k",
    4: "$20-25k",
    5: "$25-35k",
    6: "$35-50k",
    7: "$50-75k",
    8: "$75k+",
}

DEFAULT_LABEL_MAPS = {
    "Sex": SEX_LABELS,
    "Age": AGE_LABELS,
    "Income": INCOME_LABELS,
}


def _convert_numpy(obj: Any) -> Any:
    """Helper to convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(x) for x in obj]
    return obj


def compute_group_metrics(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_pred: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Optional[Union[pd.Series, np.ndarray, List[float]]],
    group_values: Union[pd.Series, np.ndarray, List[Any]],
    label_map: Optional[Dict[Any, str]] = None,
) -> pd.DataFrame:
    """
    Calculates subgroup metrics: TPR, FPR, FNR, selection rate, accuracy, and sample count.

    Parameters
    ----------
    y_true : pd.Series, np.ndarray, or List[int]
        Ground truth binary labels (0 or 1).
    y_pred : pd.Series, np.ndarray, or List[int]
        Predicted binary labels (0 or 1).
    y_prob : pd.Series, np.ndarray, or List[float], optional
        Predicted positive class probabilities.
    group_values : pd.Series, np.ndarray, or List[Any]
        Subgroup membership identifiers for each instance.
    label_map : dict, optional
        Mapping from group code/value to human-readable label string.

    Returns
    -------
    pd.DataFrame
        Table containing subgroup performance metrics.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    y_pr = np.asarray(y_prob).astype(float) if y_prob is not None else None
    groups = np.asarray(group_values)

    unique_groups = np.unique(groups)
    # Sort groups numerically or alphabetically
    try:
        unique_groups = np.sort(unique_groups)
    except Exception:
        pass

    rows = []
    for g in unique_groups:
        mask = (groups == g)
        n = int(np.sum(mask))
        if n == 0:
            continue

        yt_g = y_t[mask]
        yp_g = y_p[mask]

        tp = int(np.sum((yt_g == 1) & (yp_g == 1)))
        fp = int(np.sum((yt_g == 0) & (yp_g == 1)))
        tn = int(np.sum((yt_g == 0) & (yp_g == 0)))
        fn = int(np.sum((yt_g == 1) & (yp_g == 0)))

        positives = tp + fn
        negatives = tn + fp

        selection_rate = float(np.mean(yp_g == 1))
        tpr = float(tp / positives) if positives > 0 else 0.0
        fpr = float(fp / negatives) if negatives > 0 else 0.0
        fnr = float(fn / positives) if positives > 0 else 0.0
        accuracy = float((tp + tn) / n)

        mean_prob = float(np.mean(y_pr[mask])) if y_pr is not None else float("nan")

        g_code = int(g) if isinstance(g, (np.integer, int)) or (isinstance(g, (float, np.floating)) and g.is_integer()) else g
        g_name = label_map.get(g_code, str(g_code)) if label_map else str(g_code)

        rows.append({
            "group_code": g_code,
            "group_name": g_name,
            "n": n,
            "selection_rate": round(selection_rate, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "accuracy": round(accuracy, 4),
            "mean_predicted_prob": round(mean_prob, 4) if not np.isnan(mean_prob) else None,
        })

    return pd.DataFrame(rows)


def compute_fairness_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]],
    X_test_df: pd.DataFrame,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes comprehensive fairness metrics across Sex, Age, and Income.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary labels.
    y_pred : pd.Series or np.ndarray
        Predicted binary labels.
    y_prob : pd.Series or np.ndarray, optional
        Predicted positive class probabilities.
    X_test_df : pd.DataFrame
        Test set features containing columns 'Sex', 'Age', 'Income'.
    model_name : str, default='Model'
        Name of the model being evaluated.

    Returns
    -------
    dict
        Dictionary containing subgroup metrics and disparity metrics for each attribute.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    y_pr = np.asarray(y_prob).astype(float) if y_prob is not None else None

    attributes_dict = {}

    target_attributes = [col for col in ["Sex", "Age", "Income"] if col in X_test_df.columns]

    for attr in target_attributes:
        group_col = X_test_df[attr].values
        label_map = DEFAULT_LABEL_MAPS.get(attr, None)

        group_df = compute_group_metrics(y_t, y_p, y_pr, group_col, label_map=label_map)

        sel_rates = group_df["selection_rate"].values
        tprs = group_df["tpr"].values
        fprs = group_df["fpr"].values

        demographic_parity_diff = float(np.max(sel_rates) - np.min(sel_rates)) if len(sel_rates) > 0 else 0.0
        max_tpr_diff = float(np.max(tprs) - np.min(tprs)) if len(tprs) > 0 else 0.0
        max_fpr_diff = float(np.max(fprs) - np.min(fprs)) if len(fprs) > 0 else 0.0
        equalized_odds_diff = float(max(max_tpr_diff, max_fpr_diff))

        attributes_dict[attr] = {
            "group_metrics": group_df.to_dict(orient="records"),
            "demographic_parity_difference": round(demographic_parity_diff, 4),
            "equalized_odds_difference": round(equalized_odds_diff, 4),
            "max_tpr_difference": round(max_tpr_diff, 4),
            "max_fpr_difference": round(max_fpr_diff, 4),
        }

    return {
        "model_name": model_name,
        "attributes": attributes_dict,
    }


def compute_fairness_both_models(
    baseline_preds: Union[pd.Series, np.ndarray],
    trustworthy_preds: Union[pd.Series, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
    y_prob_base: Optional[Union[pd.Series, np.ndarray]],
    y_prob_trust: Optional[Union[pd.Series, np.ndarray]],
    X_test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Runs fairness evaluation on both models and returns combined dictionary.

    Parameters
    ----------
    baseline_preds : array-like
        Binary predictions from baseline model.
    trustworthy_preds : array-like
        Binary predictions from trustworthy model.
    y_true : array-like
        True binary labels.
    y_prob_base : array-like, optional
        Predicted probabilities from baseline model.
    y_prob_trust : array-like, optional
        Predicted probabilities from trustworthy model.
    X_test_df : pd.DataFrame
        Test DataFrame with demographic columns.

    Returns
    -------
    dict
        Combined fairness metrics for both models.
    """
    baseline_fairness = compute_fairness_metrics(
        y_true=y_true,
        y_pred=baseline_preds,
        y_prob=y_prob_base,
        X_test_df=X_test_df,
        model_name="Baseline (Model 1)",
    )

    trustworthy_fairness = compute_fairness_metrics(
        y_true=y_true,
        y_pred=trustworthy_preds,
        y_prob=y_prob_trust,
        X_test_df=X_test_df,
        model_name="Trust-Aware (Model 2)",
    )

    # Build comparative summary
    comparison_summary = {}
    for attr in baseline_fairness["attributes"].keys():
        b_dp = baseline_fairness["attributes"][attr]["demographic_parity_difference"]
        t_dp = trustworthy_fairness["attributes"][attr]["demographic_parity_difference"]
        b_eo = baseline_fairness["attributes"][attr]["equalized_odds_difference"]
        t_eo = trustworthy_fairness["attributes"][attr]["equalized_odds_difference"]

        comparison_summary[attr] = {
            "demographic_parity_diff_baseline": b_dp,
            "demographic_parity_diff_trustworthy": t_dp,
            "demographic_parity_change": round(t_dp - b_dp, 4),
            "equalized_odds_diff_baseline": b_eo,
            "equalized_odds_diff_trustworthy": t_eo,
            "equalized_odds_change": round(t_eo - b_eo, 4),
        }

    return {
        "baseline_model": baseline_fairness,
        "trustworthy_model": trustworthy_fairness,
        "comparison_summary": comparison_summary,
    }


def save_fairness_artifacts(
    fairness_dict: Dict[str, Any],
    artifacts_dir: str = "artifacts",
) -> str:
    """
    Saves fairness metrics dictionary to artifacts/fairness_metrics.json.

    Parameters
    ----------
    fairness_dict : dict
        Combined fairness metrics dictionary.
    artifacts_dir : str, default='artifacts'
        Target directory.

    Returns
    -------
    str
        Path to saved JSON artifact.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "fairness_metrics.json")
    clean_data = _convert_numpy(fairness_dict)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)

    return out_path
