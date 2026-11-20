"""
Data loading, validation, dictionary definitions, and splitting module.

This module provides functions to:
- Load and validate the BRFSS 2015 diabetes dataset.
- Check target correlations and feature validity ranges.
- Remove duplicate rows.
- Perform stratified train/validation/test splits (70% / 15% / 15%).
- Provide metadata, feature groupings, and human-interpretable data dictionaries.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_feature_names() -> List[str]:
    """
    Return the ordered list of 21 feature column names (excluding target).

    Returns:
        List[str]: List of 21 predictor feature names.
    """
    return [
        'HighBP',
        'HighChol',
        'CholCheck',
        'BMI',
        'Smoker',
        'Stroke',
        'HeartDiseaseorAttack',
        'PhysActivity',
        'Fruits',
        'Veggies',
        'HvyAlcoholConsump',
        'AnyHealthcare',
        'NoDocbcCost',
        'GenHlth',
        'MentHlth',
        'PhysHlth',
        'DiffWalk',
        'Sex',
        'Age',
        'Education',
        'Income',
    ]


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return feature names grouped by measurement type: continuous, ordinal, and binary.

    Returns:
        Dict[str, List[str]]: Dictionary with keys 'continuous', 'ordinal', 'binary'.
    """
    return {
        'continuous': ['BMI', 'MentHlth', 'PhysHlth'],
        'ordinal': ['GenHlth', 'Age', 'Education', 'Income'],
        'binary': [
            'HighBP',
            'HighChol',
            'CholCheck',
            'Smoker',
            'Stroke',
            'HeartDiseaseorAttack',
            'PhysActivity',
            'Fruits',
            'Veggies',
            'HvyAlcoholConsump',
            'AnyHealthcare',
            'NoDocbcCost',
            'DiffWalk',
            'Sex',
        ],
    }


def get_data_dictionary() -> Dict[str, Dict[str, Union[str, Optional[Dict[int, str]]]]]:
    """
    Return comprehensive BRFSS data dictionary with descriptions, types, valid ranges, and label maps.

    Returns:
        Dict[str, Dict[str, Union[str, Optional[Dict[int, str]]]]]:
            Mapping from feature name to dictionary of metadata.
    """
    binary_map = {0: 'No', 1: 'Yes'}
    sex_map = {0: 'Female', 1: 'Male'}
    gen_hlth_map = {
        1: 'Excellent',
        2: 'Very good',
        3: 'Good',
        4: 'Fair',
        5: 'Poor',
    }
    age_map = {
        1: '18-24',
        2: '25-29',
        3: '30-34',
        4: '35-39',
        5: '40-44',
        6: '45-49',
        7: '50-54',
        8: '55-59',
        9: '60-64',
        10: '65-69',
        11: '70-74',
        12: '75-79',
        13: '80+',
    }
    education_map = {
        1: 'Never attended',
        2: 'Grades 1-8',
        3: 'Grades 9-11',
        4: 'Grade 12/GED',
        5: 'College 1-3 yrs',
        6: 'College 4+ yrs',
    }
    income_map = {
        1: '<$10k',
        2: '$10-15k',
        3: '$15-20k',
        4: '$20-25k',
        5: '$25-35k',
        6: '$35-50k',
        7: '$50-75k',
        8: '$75k+',
    }

    return {
        'HighBP': {
            'description': 'Adults told they have high blood pressure by a health professional',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'HighChol': {
            'description': 'Adults told they have high blood cholesterol',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'CholCheck': {
            'description': 'Cholesterol check within past five years',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'BMI': {
            'description': 'Body Mass Index (BMI)',
            'type': 'continuous',
            'valid_range': '12-98',
            'label_map': None,
        },
        'Smoker': {
            'description': 'Smoked at least 100 cigarettes in lifetime',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'Stroke': {
            'description': 'Ever diagnosed with a stroke',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'HeartDiseaseorAttack': {
            'description': 'Coronary heart disease (CHD) or myocardial infarction (MI)',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'PhysActivity': {
            'description': 'Physical activity or exercise in past 30 days other than regular job',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'Fruits': {
            'description': 'Consume fruit 1 or more times per day',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'Veggies': {
            'description': 'Consume vegetables 1 or more times per day',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'HvyAlcoholConsump': {
            'description': 'Heavy alcohol consumption (adult men >14 drinks/week, adult women >7 drinks/week)',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'AnyHealthcare': {
            'description': 'Any kind of health care coverage or insurance',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'NoDocbcCost': {
            'description': 'Needed to see a doctor in past 12 months but could not due to cost',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'GenHlth': {
            'description': 'Self-reported general health status (1=Excellent, 5=Poor)',
            'type': 'ordinal',
            'valid_range': '1-5',
            'label_map': gen_hlth_map,
        },
        'MentHlth': {
            'description': 'Days in past 30 days mental health not good (stress, depression, emotional problems)',
            'type': 'continuous',
            'valid_range': '0-30',
            'label_map': None,
        },
        'PhysHlth': {
            'description': 'Days in past 30 days physical health not good (illness and injury)',
            'type': 'continuous',
            'valid_range': '0-30',
            'label_map': None,
        },
        'DiffWalk': {
            'description': 'Serious difficulty walking or climbing stairs',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': binary_map,
        },
        'Sex': {
            'description': 'Reported biological sex (0=Female, 1=Male)',
            'type': 'binary',
            'valid_range': '0-1',
            'label_map': sex_map,
        },
        'Age': {
            'description': 'Thirteen-level age category (1=18-24, 13=80+)',
            'type': 'ordinal',
            'valid_range': '1-13',
            'label_map': age_map,
        },
        'Education': {
            'description': 'Highest level of education completed (1=Never attended, 6=College 4+ yrs)',
            'type': 'ordinal',
            'valid_range': '1-6',
            'label_map': education_map,
        },
        'Income': {
            'description': 'Household income level (1=<$10k, 8=$75k+)',
            'type': 'ordinal',
            'valid_range': '1-8',
            'label_map': income_map,
        },
    }


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load raw CSV dataset, print dataset summary, and validate integrity and ranges.

    Validation checks:
    - Verifies file exists and loads into a DataFrame.
    - Prints shape, data types, missing value count, duplicate count, target distribution.
    - Validates feature ranges against BRFSS specifications.
    - Confirms target correlation does not exceed 0.95 (leakage prevention).

    Args:
        csv_path: Path to the raw CSV dataset.

    Returns:
        pd.DataFrame: Full raw dataset.
    """
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset path: {csv_path}")
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Missing values check
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total missing values: {total_missing}")
    if total_missing > 0:
        print("Missing values per column:\n", missing_counts[missing_counts > 0])

    # Duplicates check
    n_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {n_duplicates:,} ({n_duplicates / len(df) * 100:.2f}%)")

    # Target distribution
    target_col = 'Diabetes_binary'
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().to_dict()
        target_props = (df[target_col].value_counts(normalize=True) * 100).to_dict()
        print(f"Target distribution ('{target_col}'):")
        for cls, count in sorted(target_counts.items()):
            print(f"  Class {cls}: {count:,} ({target_props[cls]:.2f}%)")
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # Range validation
    expected_ranges = {
        'Diabetes_binary': (0, 1),
        'HighBP': (0, 1),
        'HighChol': (0, 1),
        'CholCheck': (0, 1),
        'BMI': (12, 98),
        'Smoker': (0, 1),
        'Stroke': (0, 1),
        'HeartDiseaseorAttack': (0, 1),
        'PhysActivity': (0, 1),
        'Fruits': (0, 1),
        'Veggies': (0, 1),
        'HvyAlcoholConsump': (0, 1),
        'AnyHealthcare': (0, 1),
        'NoDocbcCost': (0, 1),
        'GenHlth': (1, 5),
        'MentHlth': (0, 30),
        'PhysHlth': (0, 30),
        'DiffWalk': (0, 1),
        'Sex': (0, 1),
        'Age': (1, 13),
        'Education': (1, 6),
        'Income': (1, 8),
    }

    for col, (min_val, max_val) in expected_ranges.items():
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            if actual_min < min_val or actual_max > max_val:
                raise ValueError(
                    f"Column '{col}' has values outside valid range [{min_val}, {max_val}]: "
                    f"actual range is [{actual_min}, {actual_max}]"
                )

    print("All feature value ranges successfully validated against BRFSS codebook specifications.")

    # Target leakage check: ensure no predictor has correlation > 0.95 with target
    correlations = df.corr()[target_col].drop(target_col).abs()
    max_corr = correlations.max()
    max_corr_feature = correlations.idxmax()
    print(f"Maximum feature correlation with '{target_col}': {max_corr:.4f} ('{max_corr_feature}')")
    assert max_corr < 0.95, (
        f"Potential target leakage detected: feature '{max_corr_feature}' has correlation {max_corr:.4f} >= 0.95"
    )
    print("Leakage check passed: No feature exceeds 0.95 correlation with target.")
    print("=" * 60)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from DataFrame and report before/after row counts.

    Args:
        df: Input pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    n_before = len(df)
    n_dupes = df.duplicated().sum()
    clean_df = df.drop_duplicates().copy()
    n_after = len(clean_df)

    print(
        f"Duplicate Removal: {n_dupes:,} exact duplicate rows removed. "
        f"Row count changed from {n_before:,} to {n_after:,}."
    )
    return clean_df


def split_dataset(df: pd.DataFrame, random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]
]:
    """
    Split dataset into stratified Train (70%), Validation (15%), and Test (15%) sets.

    Stratification is performed on the binary target 'Diabetes_binary'.
    A two-step stratified split is used:
    1. 70% Train, 30% Temporary holdout.
    2. Split Temporary holdout 50/50 into Validation (15%) and Test (15%).

    Args:
        df: Input pandas DataFrame containing all features and 'Diabetes_binary'.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing:
            X_train (pd.DataFrame): Training features (70%).
            X_val (pd.DataFrame): Validation features (15%).
            X_test (pd.DataFrame): Test features (15%).
            y_train (pd.Series): Training target (70%).
            y_val (pd.Series): Validation target (15%).
            y_test (pd.Series): Test target (15%).
            feature_names (List[str]): List of 21 feature column names.
    """
    target_col = 'Diabetes_binary'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Step 1: 70% Train, 30% Holdout (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Step 2: 50% of Holdout = 15% Val, 50% of Holdout = 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    print("=" * 60)
    print("STRATIFIED DATA SPLIT SUMMARY (70% Train / 15% Val / 15% Test)")
    print("=" * 60)
    total_n = len(df)
    splits = [
        ('Train', X_train, y_train),
        ('Validation', X_val, y_val),
        ('Test', X_test, y_test),
    ]

    for name, X_s, y_s in splits:
        pos_n = int((y_s == 1).sum())
        neg_n = int((y_s == 0).sum())
        pos_rate = (y_s == 1).mean() * 100
        print(
            f"  {name:12}: {len(X_s):6,} rows ({len(X_s) / total_n * 100:.1f}%) | "
            f"Class 0: {neg_n:5,} | Class 1: {pos_n:5,} | Prev: {pos_rate:.2f}%"
        )
    print(f"Features: {len(feature_names)} columns ({feature_names})")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
