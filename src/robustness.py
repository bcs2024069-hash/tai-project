"""
Robustness and Perturbation Testing Module for Diabetes Risk-Screening.

Tests model sensitivity to controlled single-feature perturbations strictly within
valid codebook clinical ranges:
- bmi_plus1 (+1, max 98)
- bmi_minus1 (-1, min 12)
- genhlth_worse (+1, max 5)
- genhlth_better (-1, min 1)
- physhlth_plus1 (+1, max 30)
- menthlth_plus1 (+1, max 30)

Evaluates whether small clinical shifts flip risk classifications or induce excessive
probability instability across models.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


PERTURBATION_CONFIGS = {
    "bmi_plus1": {
        "feature": "BMI",
        "delta": 1.0,
        "min_val": 12.0,
        "max_val": 98.0,
        "description": "Increase BMI by 1 unit (capped at 98)",
    },
    "bmi_minus1": {
        "feature": "BMI",
        "delta": -1.0,
        "min_val": 12.0,
        "max_val": 98.0,
        "description": "Decrease BMI by 1 unit (floored at 12)",
    },
    "genhlth_worse": {
        "feature": "GenHlth",
        "delta": 1.0,
        "min_val": 1.0,
        "max_val": 5.0,
        "description": "Worsen general health status by 1 point (capped at 5)",
    },
    "genhlth_better": {
        "feature": "GenHlth",
        "delta": -1.0,
        "min_val": 1.0,
        "max_val": 5.0,
        "description": "Improve general health status by 1 point (floored at 1)",
    },
    "physhlth_plus1": {
        "feature": "PhysHlth",
        "delta": 1.0,
        "min_val": 0.0,
        "max_val": 30.0,
        "description": "Add 1 physical health injury/illness day (capped at 30)",
    },
    "menthlth_plus1": {
        "feature": "MentHlth",
        "delta": 1.0,
        "min_val": 0.0,
        "max_val": 30.0,
        "description": "Add 1 poor mental health day (capped at 30)",
    },
}


def apply_perturbation(X_df: pd.DataFrame, perturbation_name: str) -> pd.DataFrame:
    """
    Creates a perturbed copy of the feature DataFrame within strictly valid ranges.

    Parameters
    ----------
    X_df : pd.DataFrame
        Input features.
    perturbation_name : str
        Name of perturbation (key in PERTURBATION_CONFIGS).

    Returns
    -------
    pd.DataFrame
        Copy of X_df with the targeted feature perturbed and clamped.
    """
    if perturbation_name not in PERTURBATION_CONFIGS:
        raise ValueError(
            f"Unknown perturbation '{perturbation_name}'. Available: {list(PERTURBATION_CONFIGS.keys())}"
        )

    cfg = PERTURBATION_CONFIGS[perturbation_name]
    feature = cfg["feature"]

    if feature not in X_df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns.")

    X_pert = X_df.copy()
    X_pert[feature] = (X_pert[feature] + cfg["delta"]).clip(
        lower=cfg["min_val"], upper=cfg["max_val"]
    )
    return X_pert


def run_robustness_test(
    model,
    X_test_df: pd.DataFrame,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    perturbation_name: str = "bmi_plus1",
    n_samples: int = 500,
    random_state: int = 42,
    sample_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Measures probability change and classification flips for a single perturbation.

    Parameters
    ----------
    model : Pipeline or CalibratedClassifierCV
        Trained model with predict_proba.
    X_test_df : pd.DataFrame
        Test features.
    y_test : array-like, optional
        True labels (optional).
    perturbation_name : str
        Perturbation identifier.
    n_samples : int, default=500
        Number of test cases to sample if sample_indices is not provided.
    random_state : int, default=42
        Random seed for sampling.
    sample_indices : np.ndarray, optional
        Fixed row indices to evaluate (ensures exact comparison across runs).

    Returns
    -------
    pd.DataFrame
        Individual test case results with original and perturbed probabilities.
    """
    total_n = len(X_test_df)
    if sample_indices is None:
        rng = np.random.RandomState(random_state)
        sample_size = min(n_samples, total_n)
        sample_indices = rng.choice(total_n, size=sample_size, replace=False)

    X_sample = X_test_df.iloc[sample_indices].copy()
    X_perturbed = apply_perturbation(X_sample, perturbation_name)

    orig_probs = model.predict_proba(X_sample)[:, 1]
    pert_probs = model.predict_proba(X_perturbed)[:, 1]

    abs_prob_change = np.abs(pert_probs - orig_probs)
    flipped = ((orig_probs >= 0.5) != (pert_probs >= 0.5)).astype(int)

    results = pd.DataFrame({
        "sample_index": sample_indices,
        "perturbation": perturbation_name,
        "original_prob": np.round(orig_probs, 4),
        "perturbed_prob": np.round(pert_probs, 4),
        "abs_prob_change": np.round(abs_prob_change, 4),
        "flipped": flipped,
    })

    return results


def run_all_perturbations(
    baseline_model,
    trustworthy_model,
    X_test_df: pd.DataFrame,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs all 6 perturbations on both models using the exact same sample instances.

    Parameters
    ----------
    baseline_model : object
        Baseline model pipeline.
    trustworthy_model : object
        Trust-aware calibrated model.
    X_test_df : pd.DataFrame
        Test features.
    y_test : array-like, optional
        Ground truth labels.
    n_samples : int, default=500
        Sample size.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        Combined results for both models across all perturbations.
    """
    total_n = len(X_test_df)
    rng = np.random.RandomState(random_state)
    sample_size = min(n_samples, total_n)
    fixed_indices = rng.choice(total_n, size=sample_size, replace=False)

    all_dfs = []

    for pert_name in PERTURBATION_CONFIGS.keys():
        # Baseline model
        df_base = run_robustness_test(
            model=baseline_model,
            X_test_df=X_test_df,
            y_test=y_test,
            perturbation_name=pert_name,
            sample_indices=fixed_indices,
        )
        df_base["model"] = "Baseline"
        all_dfs.append(df_base)

        # Trustworthy model
        df_trust = run_robustness_test(
            model=trustworthy_model,
            X_test_df=X_test_df,
            y_test=y_test,
            perturbation_name=pert_name,
            sample_indices=fixed_indices,
        )
        df_trust["model"] = "Trust-Aware"
        all_dfs.append(df_trust)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def compute_robustness_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates robustness results into a summary table by model and perturbation.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_all_perturbations or run_robustness_test.

    Returns
    -------
    pd.DataFrame
        Summary table with mean_abs_prob_change, max_abs_prob_change, flip_rate, flip_count, n_samples.
    """
    group_cols = ["model", "perturbation"] if "model" in results_df.columns else ["perturbation"]

    summary = results_df.groupby(group_cols).agg(
        mean_abs_prob_change=("abs_prob_change", "mean"),
        max_abs_prob_change=("abs_prob_change", "max"),
        flip_rate=("flipped", "mean"),
        flip_count=("flipped", "sum"),
        n_samples=("flipped", "count"),
    ).reset_index()

    summary["mean_abs_prob_change"] = summary["mean_abs_prob_change"].round(4)
    summary["max_abs_prob_change"] = summary["max_abs_prob_change"].round(4)
    summary["flip_rate"] = summary["flip_rate"].round(4)

    return summary


def save_robustness_artifacts(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> Tuple[str, str]:
    """
    Saves robustness results and summary tables to artifacts directory.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full individual test instances.
    summary_df : pd.DataFrame
        Aggregated summary table.
    artifacts_dir : str, default='artifacts'
        Target directory.

    Returns
    -------
    results_path : str
        Path to artifacts/robustness_results.csv.
    summary_path : str
        Path to artifacts/robustness_summary.csv.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    results_path = os.path.join(artifacts_dir, "robustness_results.csv")
    summary_path = os.path.join(artifacts_dir, "robustness_summary.csv")

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return results_path, summary_path
