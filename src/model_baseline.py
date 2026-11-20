"""Trustworthy Diabetes Risk-Screening - Baseline Model Module.

Implements training, evaluation, and reporting for Model 1:
Uncalibrated Baseline Logistic Regression Pipeline.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

try:
    from src.preprocessing import build_baseline_pipeline
except ImportError:
    from preprocessing import build_baseline_pipeline


def train_baseline_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    random_state: int = 42,
) -> Pipeline:
    """Build and fit the baseline logistic regression pipeline.

    Args:
        X_train: Training feature set.
        y_train: Training labels (0: No Diabetes, 1: Diabetes/Prediabetes).
        random_state: Random state for reproducibility.

    Returns:
        Fitted sklearn Pipeline containing ColumnTransformer and LogisticRegression.
    """
    pipeline = build_baseline_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
    model_name: str = "Model 1 (Baseline)",
) -> Dict[str, Any]:
    """Compute comprehensive performance and discrimination metrics for a binary classification model.

    Args:
        model: Fitted estimator or pipeline with predict_proba.
        X_test: Test features.
        y_test: Ground truth labels.
        threshold: Decision threshold for positive class classification (default 0.5).
        model_name: Descriptive name of the model being evaluated.

    Returns:
        Dictionary containing scalar metrics, raw predictions, probabilities,
        confusion matrix, and ROC curve coordinates.
    """
    y_true = np.asarray(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fpr_list = fpr.tolist()
    tpr_list = tpr.tolist()

    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "fpr": fpr_list,
        "tpr": tpr_list,
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def get_classification_report(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> str:
    """Generate scikit-learn text classification report for test dataset.

    Args:
        model: Fitted estimator or pipeline.
        X_test: Test features.
        y_test: Ground truth labels.

    Returns:
        Formatted classification report string.
    """
    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)
    return classification_report(
        y_true,
        y_pred,
        target_names=["No Diabetes", "Prediabetes/Diabetes"],
    )
