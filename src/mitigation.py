"""
Fairness Mitigation Experiment Module.

Conducts an isolated, experimental fairness post-processing study using Fairlearn's
ThresholdOptimizer. The main production models remain unaltered; this experiment evaluates
the feasibility and trade-offs of optimizing equalized odds across socio-economic groups
(specifically Income) at the expense of slight utility degradation.

Methodology:
- Uses prefit=True so the underlying classifier is NOT retrained on validation data.
- Fits group thresholds using validation data only (X_val, y_val, X_val_df['Income']).
- Evaluates using untouched test data (X_test, y_test, X_test_df['Income']).
- Supplies sensitive attribute (Income) during inference: optimizer.predict(X_test, sensitive_features=A_test, random_state=random_state).
- Operational Limitation: Group-specific thresholds strictly require sensitive attribute (Income) information at inference time.
- Underlying Baseline and Calibrated models are NOT modified.
- Reports comprehensive before-and-after fairness metrics and accuracy trade-offs.

- Fitted strictly on validation data (X_val, y_val, X_val_df['Income']).
- Evaluated strictly on held-out test data (X_test, y_test, X_test_df['Income']).
- Underlying Baseline and Calibrated models are NOT modified.
- Reports comprehensive before-and-after fairness metrics and accuracy trade-offs.

Constraints: equalized_odds
Objective: balanced_accuracy_score
Sensitive attribute: Income
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from fairlearn.postprocessing import ThresholdOptimizer

from src.fairness import DEFAULT_LABEL_MAPS, compute_group_metrics


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Renders a pandas DataFrame as a Markdown table without third-party dependencies."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join([":---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    return "\n".join([header, sep] + rows)


def run_mitigation_experiment(
    baseline_model,
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    X_val_df: pd.DataFrame,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    X_test_df: pd.DataFrame,
    sensitive_col: str = "Income",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes an isolated fairness mitigation experiment using ThresholdOptimizer.

    Strict Leakage Prevention:
    - The optimizer is fitted using validation data only (X_val, y_val, X_val_df).
    - Performance and disparity metrics are evaluated using untouched test data (X_test, y_test, X_test_df).
    - The underlying baseline model is NOT modified.

    Parameters
    ----------
    baseline_model : Pipeline
        Fitted baseline pipeline (includes preprocessor and classifier).
    X_val : pd.DataFrame or np.ndarray
        Validation features for fitting the threshold optimizer.
    y_val : pd.Series or np.ndarray
        Validation ground truth labels.
    X_val_df : pd.DataFrame
        Validation features as DataFrame containing sensitive_col.
    X_test : pd.DataFrame or np.ndarray
        Test features for final evaluation.
    y_test : pd.Series or np.ndarray
        Test ground truth labels.
    X_test_df : pd.DataFrame
        Test features as DataFrame containing sensitive_col.
    sensitive_col : str, default='Income'
        Column name representing the sensitive attribute.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Comparative dictionary containing baseline metrics, mitigated metrics,
        disparities, delta trade-offs, and group-level performance.
    """
    y_t = np.asarray(y_test).astype(int)
    y_v = np.asarray(y_val).astype(int)

    if sensitive_col not in X_val_df.columns:
        raise KeyError(f"Sensitive column '{sensitive_col}' not found in validation DataFrame.")
    if sensitive_col not in X_test_df.columns:
        raise KeyError(f"Sensitive column '{sensitive_col}' not found in test DataFrame.")

    A_val = X_val_df[sensitive_col].values
    A_test = X_test_df[sensitive_col].values
    label_map = DEFAULT_LABEL_MAPS.get(sensitive_col, None)

    # 1. Baseline Model Predictions on Held-out Test Data
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob_base >= 0.5).astype(int)

    base_acc = float(accuracy_score(y_t, y_pred_base))
    base_bal_acc = float(balanced_accuracy_score(y_t, y_pred_base))
    base_f1 = float(f1_score(y_t, y_pred_base, zero_division=0))
    base_prec = float(precision_score(y_t, y_pred_base, zero_division=0))
    base_rec = float(recall_score(y_t, y_pred_base, zero_division=0))

    base_groups_df = compute_group_metrics(
        y_true=y_t,
        y_pred=y_pred_base,
        y_prob=y_prob_base,
        group_values=A_test,
        label_map=label_map,
    )

    base_sel_rates = base_groups_df["selection_rate"].values
    base_tprs = base_groups_df["tpr"].values
    base_fprs = base_groups_df["fpr"].values

    base_dp_diff = float(np.max(base_sel_rates) - np.min(base_sel_rates))
    base_max_tpr_diff = float(np.max(base_tprs) - np.min(base_tprs))
    base_max_fpr_diff = float(np.max(base_fprs) - np.min(base_fprs))
    base_eo_diff = float(max(base_max_tpr_diff, base_max_fpr_diff))

    # 2. Fit ThresholdOptimizer strictly on Validation Data
    optimizer = ThresholdOptimizer(
        estimator=baseline_model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(X_val, y_v, sensitive_features=A_val)

    # 3. Evaluate Mitigated Model strictly on Untouched Test Data
    y_pred_mit = optimizer.predict(X_test, sensitive_features=A_test, random_state=random_state)
    y_pred_mit = np.asarray(y_pred_mit).astype(int)

    mit_acc = float(accuracy_score(y_t, y_pred_mit))
    mit_bal_acc = float(balanced_accuracy_score(y_t, y_pred_mit))
    mit_f1 = float(f1_score(y_t, y_pred_mit, zero_division=0))
    mit_prec = float(precision_score(y_t, y_pred_mit, zero_division=0))
    mit_rec = float(recall_score(y_t, y_pred_mit, zero_division=0))

    mit_groups_df = compute_group_metrics(
        y_true=y_t,
        y_pred=y_pred_mit,
        y_prob=None,
        group_values=A_test,
        label_map=label_map,
    )

    mit_sel_rates = mit_groups_df["selection_rate"].values
    mit_tprs = mit_groups_df["tpr"].values
    mit_fprs = mit_groups_df["fpr"].values

    mit_dp_diff = float(np.max(mit_sel_rates) - np.min(mit_sel_rates))
    mit_max_tpr_diff = float(np.max(mit_tprs) - np.min(mit_tprs))
    mit_max_fpr_diff = float(np.max(mit_fprs) - np.min(mit_fprs))
    mit_eo_diff = float(max(mit_max_tpr_diff, mit_max_fpr_diff))

    delta_dict = {
        "accuracy_change": round(mit_acc - base_acc, 4),
        "balanced_accuracy_change": round(mit_bal_acc - base_bal_acc, 4),
        "f1_change": round(mit_f1 - base_f1, 4),
        "demographic_parity_diff_change": round(mit_dp_diff - base_dp_diff, 4),
        "equalized_odds_diff_change": round(mit_eo_diff - base_eo_diff, 4),
    }

    results = {
        "experiment_name": "ThresholdOptimizer Equalized Odds Mitigation",
        "sensitive_attribute": sensitive_col,
        "constraint": "equalized_odds",
        "objective": "balanced_accuracy_score",
        "fitted_on": "validation_split_only",
        "evaluated_on": "untouched_test_split",
        "baseline": {
            "accuracy": round(base_acc, 4),
            "balanced_accuracy": round(base_bal_acc, 4),
            "f1": round(base_f1, 4),
            "precision": round(base_prec, 4),
            "recall": round(base_rec, 4),
            "demographic_parity_difference": round(base_dp_diff, 4),
            "equalized_odds_difference": round(base_eo_diff, 4),
            "max_tpr_difference": round(base_max_tpr_diff, 4),
            "max_fpr_difference": round(base_max_fpr_diff, 4),
            "group_metrics": base_groups_df.to_dict(orient="records"),
        },
        "mitigated": {
            "accuracy": round(mit_acc, 4),
            "balanced_accuracy": round(mit_bal_acc, 4),
            "f1": round(mit_f1, 4),
            "precision": round(mit_prec, 4),
            "recall": round(mit_rec, 4),
            "demographic_parity_difference": round(mit_dp_diff, 4),
            "equalized_odds_difference": round(mit_eo_diff, 4),
            "max_tpr_difference": round(mit_max_tpr_diff, 4),
            "max_fpr_difference": round(mit_max_fpr_diff, 4),
            "group_metrics": mit_groups_df.to_dict(orient="records"),
        },
        "deltas": delta_dict,
    }

    return results


def format_mitigation_report(mitigation_dict: Dict[str, Any]) -> str:
    """Formats the mitigation experimental findings into a structured markdown report."""
    sens = mitigation_dict["sensitive_attribute"]
    base = mitigation_dict["baseline"]
    mit = mitigation_dict["mitigated"]
    deltas = mitigation_dict["deltas"]

    lines = [
        f"# Fairness Mitigation Study: Threshold Optimization on '{sens}'",
        "",
        "## Executive Summary",
        "This experimental module evaluates post-processing fairness mitigation using Fairlearn's `ThresholdOptimizer`.",
        f"- **Sensitive Attribute**: `{sens}`",
        f"- **Constraint**: `{mitigation_dict['constraint']}`",
        f"- **Objective**: `{mitigation_dict['objective']}`",
        f"- **Data Protocol**: Fitted on Validation data only; Evaluated on untouched Test data.",
        "",
        "### Key Findings",
        f"- **Equalized Odds Difference**: Shifted from `{base['equalized_odds_difference']:.4f}` (Baseline) to `{mit['equalized_odds_difference']:.4f}` (Mitigated) "
        f"(`{deltas['equalized_odds_diff_change']:+.4f}`).",
        f"- **Demographic Parity Difference**: `{base['demographic_parity_difference']:.4f}` -> `{mit['demographic_parity_difference']:.4f}` "
        f"(`{deltas['demographic_parity_diff_change']:+.4f}`).",
        f"- **Accuracy Impact**: `{base['accuracy']:.4f}` -> `{mit['accuracy']:.4f}` (`{deltas['accuracy_change']:+.4f}`).",
        f"- **Balanced Accuracy Impact**: `{base['balanced_accuracy']:.4f}` -> `{mit['balanced_accuracy']:.4f}` (`{deltas['balanced_accuracy_change']:+.4f}`).",
        f"- **F1 Score Impact**: `{base['f1']:.4f}` -> `{mit['f1']:.4f}` (`{deltas['f1_change']:+.4f}`).",
        "",
        "## Performance & Disparity Comparison Table",
        "| Metric | Baseline | Mitigated (Equalized Odds) | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| Accuracy | {base['accuracy']:.4f} | {mit['accuracy']:.4f} | {deltas['accuracy_change']:+.4f} |",
        f"| Balanced Accuracy | {base['balanced_accuracy']:.4f} | {mit['balanced_accuracy']:.4f} | {deltas['balanced_accuracy_change']:+.4f} |",
        f"| F1 Score | {base['f1']:.4f} | {mit['f1']:.4f} | {deltas['f1_change']:+.4f} |",
        f"| Precision | {base['precision']:.4f} | {mit['precision']:.4f} | {mit['precision'] - base['precision']:+.4f} |",
        f"| Recall | {base['recall']:.4f} | {mit['recall']:.4f} | {mit['recall'] - base['recall']:+.4f} |",
        f"| Equalized Odds Diff | {base['equalized_odds_difference']:.4f} | {mit['equalized_odds_difference']:.4f} | {deltas['equalized_odds_diff_change']:+.4f} |",
        f"| Demographic Parity Diff | {base['demographic_parity_difference']:.4f} | {mit['demographic_parity_difference']:.4f} | {deltas['demographic_parity_diff_change']:+.4f} |",
        "",
        "## Subgroup Details Comparison",
        "### Baseline Subgroup Metrics:",
    ]

    base_groups = pd.DataFrame(base["group_metrics"])
    lines.append(_df_to_markdown_table(base_groups[["group_code", "group_name", "n", "selection_rate", "tpr", "fpr"]]))

    lines.append("\n### Mitigated Subgroup Metrics:")
    mit_groups = pd.DataFrame(mit["group_metrics"])
    lines.append(_df_to_markdown_table(mit_groups[["group_code", "group_name", "n", "selection_rate", "tpr", "fpr"]]))

    lines.extend([
        "",
        "## Clinical & Research Limitations",
        "1. **Isolated Experiment**: The primary Baseline and Trustworthy models remain unchanged.",
        "2. **Trade-off Rationale**: Applying group-specific decision thresholds post-hoc creates clinical dilemmas where identical biomarker values would lead to divergent screening recommendations based solely on income.",
        "3. **Conclusion**: Mitigating fairness solely through threshold tuning reduces disparity on paper but cannot fix root disparities present in the survey distribution.",
    ])

    return "\n".join(lines)
