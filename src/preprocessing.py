"""
Preprocessing and pipeline definition module.

This module provides functions to:
- Build the scikit-learn ColumnTransformer for feature scaling and passthrough.
- Standardize continuous and ordinal features while preserving binary indicator features.
- Define the baseline LogisticRegression pipeline.
- Define the trustworthy inner LogisticRegression pipeline (to be calibrated).
- Retrieve the exact feature ordering after ColumnTransformer transformation (crucial for SHAP).
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import get_feature_groups
except ImportError:
    from data_loader import get_feature_groups


def build_preprocessor() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for data preprocessing.

    Applies:
    - StandardScaler to 3 continuous features ('BMI', 'MentHlth', 'PhysHlth')
      and 4 ordinal features ('GenHlth', 'Age', 'Education', 'Income').
    - 'passthrough' on 14 binary indicator features.
    - remainder='drop' to ensure no unexpected columns pass through.

    Returns:
        ColumnTransformer: Unfitted preprocessor.
    """
    groups = get_feature_groups()
    scale_cols = groups['continuous'] + groups['ordinal']
    binary_cols = groups['binary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), scale_cols),
            ('binary', 'passthrough', binary_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_order_after_transform() -> List[str]:
    """
    Return the exact list of feature names in the order produced by ColumnTransformer.

    The order corresponds to:
    1. Continuous features (scaled): ['BMI', 'MentHlth', 'PhysHlth']
    2. Ordinal features (scaled): ['GenHlth', 'Age', 'Education', 'Income']
    3. Binary features (passthrough): 14 indicator features

    This exact column ordering is essential for SHAP explainability and coefficient interpretation.

    Returns:
        List[str]: 21 transformed feature names in exact column order.
    """
    groups = get_feature_groups()
    return groups['continuous'] + groups['ordinal'] + groups['binary']


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """
    Construct the baseline machine learning pipeline.

    Steps:
    1. 'preprocessor': ColumnTransformer with StandardScaler for continuous/ordinal and passthrough for binary.
    2. 'classifier': LogisticRegression with lbfgs solver, max_iter=1000, and unweighted classes.

    Args:
        random_state: Random state for deterministic solver optimization.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state,
        class_weight=None,
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])


def build_trustworthy_pipeline(random_state: int = 42) -> Pipeline:
    """
    Construct the trustworthy model pipeline structure.

    This returns the inner Pipeline (preprocessor + LogisticRegression) designed to be
    wrapped by CalibratedClassifierCV(estimator=..., method='sigmoid', cv='prefit') in model_trustworthy.py.

    Args:
        random_state: Random state for reproducibility.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline (inner estimator).
    """
    return build_baseline_pipeline(random_state=random_state)
