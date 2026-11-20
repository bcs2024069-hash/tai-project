"""Trustworthy Diabetes Risk-Screening - Trust-Aware Calibrated Model Module.

Implements Model 2: Platt Calibrated Logistic Regression Pipeline using
hold-out validation data for empirical probability calibration (sigmoid scaling)
with scikit-learn >= 1.6 FrozenEstimator architecture.

Methodology:
1. Fit the inner preprocessing + Logistic Regression pipeline on X_train, y_train.
2. Wrap the fitted pipeline using FrozenEstimator (scikit-learn >= 1.6).
3. Fit CalibratedClassifierCV(method='sigmoid') using X_val, y_val strictly.
4. Evaluate only on the untouched test set (X_test, y_test).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    try:
        from sklearn.calibration import FrozenEstimator
    except ImportError:
        FrozenEstimator = None

try:
    from src.model_baseline import evaluate_model
    from src.preprocessing import build_trustworthy_pipeline
except ImportError:
    from model_baseline import evaluate_model
    from preprocessing import build_trustworthy_pipeline


def train_trustworthy_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Build and fit the trustworthy calibrated logistic regression pipeline.

    Step 1: Fit inner preprocessing + LogisticRegression pipeline on X_train, y_train.
    Step 2: Wrap fitted pipeline using FrozenEstimator to freeze learned parameters.
    Step 3: Fit CalibratedClassifierCV(method='sigmoid') exclusively on X_val, y_val.
    Step 4: Return calibrated model for subsequent independent evaluation on test data.

    Args:
        X_train: Training features for core model fitting.
        y_train: Training labels for core model fitting.
        X_val: Independent validation features for calibration fitting only.
        y_val: Independent validation labels for calibration fitting only.
        random_state: Random state for reproducibility.

    Returns:
        Fitted CalibratedClassifierCV wrapper.
    """
    # Step 1: Fit inner pipeline on training split
    inner_pipeline = build_trustworthy_pipeline(random_state=random_state)
    inner_pipeline.fit(X_train, y_train)

    # Step 2: Wrap fitted pipeline using FrozenEstimator (scikit-learn >= 1.6)
    if FrozenEstimator is not None:
        frozen_pipeline = FrozenEstimator(inner_pipeline)
        calibrated_model = CalibratedClassifierCV(
            estimator=frozen_pipeline,
            method="sigmoid",
        )
    else:
        # Fallback for legacy environments
        calibrated_model = CalibratedClassifierCV(
            estimator=inner_pipeline,
            method="sigmoid",
            cv="prefit",
        )

    # Step 3: Fit calibration wrapper exclusively on validation data
    calibrated_model.fit(X_val, y_val)
    return calibrated_model


def evaluate_trustworthy_model(
    model: CalibratedClassifierCV,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate calibrated trustworthy model on untouched test dataset.

    Args:
        model: Fitted CalibratedClassifierCV instance.
        X_test: Untouched test features.
        y_test: Untouched ground truth labels.
        threshold: Decision threshold for positive classification (default 0.5).

    Returns:
        Dictionary of performance metrics, probabilities, and evaluation curves.
    """
    return evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name="Model 2 (Trust-Aware)",
    )


def get_inner_pipeline(calibrated_model: CalibratedClassifierCV) -> Pipeline:
    """Extract the underlying fitted sklearn Pipeline from CalibratedClassifierCV.

    Handles FrozenEstimator unwrapping for scikit-learn >= 1.6.

    Args:
        calibrated_model: Fitted CalibratedClassifierCV wrapper.

    Returns:
        Underlying Pipeline instance containing preprocessor and classifier.
    """
    first_classifier = calibrated_model.calibrated_classifiers_[0]
    estimator = first_classifier.estimator
    if hasattr(estimator, "estimator"):
        # Unwraps FrozenEstimator
        return estimator.estimator
    return estimator


def get_inner_logistic_regression(calibrated_model: CalibratedClassifierCV) -> LogisticRegression:
    """Extract the underlying trained LogisticRegression estimator.

    Args:
        calibrated_model: Fitted CalibratedClassifierCV wrapper.

    Returns:
        Fitted LogisticRegression instance from inner pipeline's 'classifier' step.
    """
    inner_pipeline = get_inner_pipeline(calibrated_model)
    return inner_pipeline.named_steps["classifier"]
