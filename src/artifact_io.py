"""Trustworthy Diabetes Risk-Screening - Artifact Input/Output Module.

Handles saving and loading of trained models, metric reports, tabular data,
metadata, and evaluation artifacts with robust type-safety for NumPy structures.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to serialize NumPy scalar and array types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure parent directory exists and return Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_model(model: Any, filename: str, artifacts_dir: str = "artifacts") -> str:
    """Save an estimator or pipeline using joblib.

    Args:
        model: Trained model or pipeline to save.
        filename: Target filename (e.g. 'baseline_model.joblib').
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved artifact.
    """
    target_dir = _ensure_dir(artifacts_dir)
    file_path = target_dir / filename
    joblib.dump(model, file_path)
    return str(file_path.resolve())


def load_model(filename: str, artifacts_dir: str = "artifacts") -> Any:
    """Load an estimator or pipeline using joblib.

    Args:
        filename: Filename to load (e.g. 'baseline_model.joblib').
        artifacts_dir: Directory containing artifact.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """
    file_path = Path(artifacts_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found at: {file_path.resolve()}")
    return joblib.load(file_path)


def save_json(data: dict, filename: str, artifacts_dir: str = "artifacts") -> str:
    """Save dictionary to a formatted JSON file with NumPy type safety.

    Args:
        data: Dictionary data to persist.
        filename: Target JSON filename.
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved JSON file.
    """
    target_dir = _ensure_dir(artifacts_dir)
    file_path = target_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return str(file_path.resolve())


def load_json(filename: str, artifacts_dir: str = "artifacts") -> dict:
    """Load JSON file into a Python dictionary.

    Args:
        filename: Filename to load.
        artifacts_dir: Directory containing artifact.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    file_path = Path(artifacts_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"JSON artifact not found at: {file_path.resolve()}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, filename: str, artifacts_dir: str = "artifacts") -> str:
    """Save pandas DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filename: Target CSV filename.
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved CSV file.
    """
    target_dir = _ensure_dir(artifacts_dir)
    file_path = target_dir / filename
    df.to_csv(file_path, index=False)
    return str(file_path.resolve())


def load_csv(filename: str, artifacts_dir: str = "artifacts") -> pd.DataFrame:
    """Load CSV into a pandas DataFrame.

    Args:
        filename: CSV filename to load.
        artifacts_dir: Directory containing artifact.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    file_path = Path(artifacts_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"CSV artifact not found at: {file_path.resolve()}")
    return pd.read_csv(file_path)


def _filter_metric_scalars(metrics_dict: dict) -> dict:
    """Filter out large prediction arrays (y_pred, y_prob, fpr, tpr) to keep metric summary."""
    excluded_keys = {"y_pred", "y_prob", "fpr", "tpr"}
    filtered = {}
    for k, v in metrics_dict.items():
        if k in excluded_keys:
            continue
        filtered[k] = v
    return filtered


def save_model_metadata(
    baseline_metrics: dict,
    trustworthy_metrics: dict,
    feature_names: List[str],
    artifacts_dir: str = "artifacts",
    extra_metadata: Optional[dict] = None,
) -> str:
    """Write model metadata, feature catalog, and metric summaries to model_metadata.json.

    Args:
        baseline_metrics: Dictionary containing baseline model evaluation metrics.
        trustworthy_metrics: Dictionary containing trustworthy model evaluation metrics.
        feature_names: List of feature names used by models.
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved model_metadata.json.
    """
    metadata = {
        "project": "Trustworthy Diabetes Risk-Screening System",
        "dataset": "CDC BRFSS 2015 (balanced 50/50)",
        "random_seed": 42,
        "feature_names": feature_names,
        "target": "Diabetes_binary",
        "target_classes": {0: "No Diabetes", 1: "Prediabetes/Diabetes"},
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "model_1": {
            "name": "Model 1 (Baseline Logistic Regression)",
            **_filter_metric_scalars(baseline_metrics),
        },
        "model_2": {
            "name": "Model 2 (Trust-Aware Calibrated LR)",
            **_filter_metric_scalars(trustworthy_metrics),
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return save_json(metadata, "model_metadata.json", artifacts_dir=artifacts_dir)


def save_evaluation_metrics(
    baseline_metrics: dict,
    trustworthy_metrics: dict,
    artifacts_dir: str = "artifacts",
) -> str:
    """Save combined evaluation metrics for both models, excluding raw sample arrays.

    Args:
        baseline_metrics: Dictionary of baseline evaluation metrics.
        trustworthy_metrics: Dictionary of trustworthy evaluation metrics.
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved evaluation_metrics.json.
    """
    combined_metrics = {
        "baseline_model": _filter_metric_scalars(baseline_metrics),
        "trustworthy_model": _filter_metric_scalars(trustworthy_metrics),
    }
    return save_json(combined_metrics, "evaluation_metrics.json", artifacts_dir=artifacts_dir)


def save_sample_cases(
    sample_cases_df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> str:
    """Save clinical persona sample cases to sample_cases.csv.

    Args:
        sample_cases_df: DataFrame containing sample cases with predictions and actuals.
        artifacts_dir: Target directory path.

    Returns:
        Full path to saved sample_cases.csv.
    """
    return save_csv(sample_cases_df, "sample_cases.csv", artifacts_dir=artifacts_dir)


def load_all_artifacts(artifacts_dir: str = "artifacts") -> Dict[str, Any]:
    """Safely loads and returns dict of all standard artifacts, returning None for missing files.

    Keys included:
        - baseline_model: loaded from baseline_model.joblib
        - trustworthy_model: loaded from trustworthy_model.joblib
        - evaluation_metrics: loaded from evaluation_metrics.json
        - fairness_metrics: loaded from fairness_metrics.json
        - calibration_data: loaded from calibration_data.csv
        - sample_cases: loaded from sample_cases.csv
        - shap_global: loaded from shap_global_importance.csv
        - shap_individual: loaded from shap_individual_values.csv
        - robustness_results: loaded from robustness_results.csv
        - robustness_summary: loaded from robustness_summary.csv
        - model_metadata: loaded from model_metadata.json

    Args:
        artifacts_dir: Directory containing artifacts.

    Returns:
        Dictionary mapping artifact identifiers to loaded objects or None if missing.
    """
    artifact_specs = {
        "baseline_model": ("baseline_model.joblib", load_model),
        "trustworthy_model": ("trustworthy_model.joblib", load_model),
        "evaluation_metrics": ("evaluation_metrics.json", load_json),
        "fairness_metrics": ("fairness_metrics.json", load_json),
        "calibration_data": ("calibration_data.csv", load_csv),
        "sample_cases": ("sample_cases.csv", load_csv),
        "shap_global": ("shap_global_importance.csv", load_csv),
        "shap_individual": ("shap_individual_values.csv", load_csv),
        "robustness_results": ("robustness_results.csv", load_csv),
        "robustness_summary": ("robustness_summary.csv", load_csv),
        "model_metadata": ("model_metadata.json", load_json),
    }

    loaded_artifacts: Dict[str, Any] = {}
    for key, (filename, loader_func) in artifact_specs.items():
        try:
            loaded_artifacts[key] = loader_func(filename, artifacts_dir=artifacts_dir)
        except FileNotFoundError:
            loaded_artifacts[key] = None
        except Exception:
            loaded_artifacts[key] = None

    return loaded_artifacts
