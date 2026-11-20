"""
Full Model Training and Trustworthy AI Artifact Generation Script.
Executes primary training on 70,692 BRFSS records, secondary sensitivity on 69,057 deduplicated records,
and exports all 11 standardized artifacts to the artifacts/ directory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import (
    data_loader,
    preprocessing,
    model_baseline,
    model_trustworthy,
    explainability,
    fairness,
    calibration,
    robustness,
    mitigation,
    artifact_io
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*75)
print("TRUSTWORTHY DIABETES RISK-SCREENING: FULL TRAINING & ARTIFACT GENERATION")
print("="*75)

# 1. LOAD PRIMARY DATASET
csv_path = os.path.join(PROJECT_ROOT, "data", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
print(f"\n[1/8] Loading and validating primary dataset from: {csv_path}")
raw_df = data_loader.load_and_validate_dataset(csv_path)
feature_names = data_loader.get_feature_names()
print(f"Primary dataset loaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns.")

# 2. PRIMARY STRATIFIED SPLIT (70% Train / 15% Val / 15% Test)
print("\n[2/8] Executing leak-free stratified split on PRIMARY dataset...")
X_train, X_val, X_test, y_train, y_val, y_test, _ = data_loader.split_dataset(raw_df, random_state=RANDOM_SEED)

X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_val_df = pd.DataFrame(X_val, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# 3. TRAIN & EVALUATE MODEL 1 (BASELINE)
print("\n[3/8] Fitting Model 1 (Baseline Logistic Regression) on 70% training split...")
baseline_model = model_baseline.train_baseline_model(X_train, y_train, random_state=RANDOM_SEED)
baseline_metrics = model_baseline.evaluate_model(
    baseline_model, X_test, y_test, threshold=0.5, model_name="Model 1 (Baseline)"
)
print(f"Model 1 Test Results -> Accuracy: {baseline_metrics['accuracy']:.4f}, AUC: {baseline_metrics['roc_auc']:.4f}, F1: {baseline_metrics['f1']:.4f}, Brier: {baseline_metrics['brier_score']:.4f}")

# 4. TRAIN & EVALUATE MODEL 2 (TRUST-AWARE WITH FROZENESTIMATOR)
print("\n[4/8] Fitting Model 2 (Trust-Aware Calibrated LR) with FrozenEstimator on validation split...")
trustworthy_model = model_trustworthy.train_trustworthy_model(
    X_train, y_train, X_val, y_val, random_state=RANDOM_SEED
)
trustworthy_metrics = model_trustworthy.evaluate_trustworthy_model(
    trustworthy_model, X_test, y_test, threshold=0.5
)
print(f"Model 2 Test Results -> Accuracy: {trustworthy_metrics['accuracy']:.4f}, AUC: {trustworthy_metrics['roc_auc']:.4f}, F1: {trustworthy_metrics['f1']:.4f}, Brier: {trustworthy_metrics['brier_score']:.4f}")

# 5. SECONDARY SENSITIVITY EXPERIMENT (DEDUPLICATED DATASET)
print("\n[5/8] Running SECONDARY sensitivity experiment on deduplicated cohort (69,057 rows)...")
clean_df = data_loader.remove_duplicates(raw_df)
X_tr_cln, X_v_cln, X_te_cln, y_tr_cln, y_v_cln, y_te_cln, _ = data_loader.split_dataset(clean_df, random_state=RANDOM_SEED)
m1_cln = model_baseline.train_baseline_model(X_tr_cln, y_tr_cln, random_state=RANDOM_SEED)
e1_cln = model_baseline.evaluate_model(m1_cln, X_te_cln, y_te_cln, threshold=0.5)
m2_cln = model_trustworthy.train_trustworthy_model(X_tr_cln, y_tr_cln, X_v_cln, y_v_cln, random_state=RANDOM_SEED)
e2_cln = model_trustworthy.evaluate_trustworthy_model(m2_cln, X_te_cln, y_te_cln, threshold=0.5)

dedup_sensitivity = {
    "secondary_dataset_rows": len(clean_df),
    "duplicates_removed": len(raw_df) - len(clean_df),
    "raw_baseline_accuracy": baseline_metrics["accuracy"],
    "dedup_baseline_accuracy": e1_cln["accuracy"],
    "accuracy_delta": round(e1_cln["accuracy"] - baseline_metrics["accuracy"], 4),
    "raw_baseline_auc": baseline_metrics["roc_auc"],
    "dedup_baseline_auc": e1_cln["roc_auc"],
    "auc_delta": round(e1_cln["roc_auc"] - baseline_metrics["roc_auc"], 4),
    "raw_baseline_brier": baseline_metrics["brier_score"],
    "dedup_baseline_brier": e1_cln["brier_score"],
    "brier_delta": round(e1_cln["brier_score"] - baseline_metrics["brier_score"], 4),
    "methodological_note": "Identical feature profiles in BRFSS reflect distinct respondents (408 profiles show conflicting diabetes labels). Deduplication does not guarantee feature-level isolation across splits; both cohorts yield virtually identical predictive performance."
}
print(f"Sensitivity Comparison: Accuracy Delta = {dedup_sensitivity['accuracy_delta']:+.4f}, AUC Delta = {dedup_sensitivity['auc_delta']:+.4f}")

# 6. TRUSTWORTHY AI AUDITS (PRIMARY TEST SET)
print("\n[6/8] Executing Trustworthy AI Audits on Held-Out Test Set...")

# 6.1 SHAP Explainability
print("  > Computing Global SHAP values (LinearExplainer on inner uncalibrated log-odds)...")
feature_order = preprocessing.get_feature_order_after_transform()
shap_values, explained_features = explainability.compute_global_shap(
    trustworthy_model,
    X_train_df,
    feature_order,
    n_background=200,
    n_explain=500,
    random_state=RANDOM_SEED
)
global_importance_df = explainability.compute_global_importance(shap_values, explained_features)
print(f"    Top 3 risk features by |SHAP|: {global_importance_df['feature'].head(3).tolist()}")

# 6.2 Sample Cases (15 representative held-out test cases)
print("  > Selecting exactly 15 stratified test cases (8 negative, 7 positive)...")
y_test_arr = np.asarray(y_test)
test_indices_0 = np.where(y_test_arr == 0)[0]
test_indices_1 = np.where(y_test_arr == 1)[0]
selected_indices = np.concatenate([test_indices_0[:8], test_indices_1[:7]])
case_ids = [f"CASE_{i+1:02d}" for i in range(15)]

sample_cases_list = []
for i, idx in enumerate(selected_indices):
    case_id = case_ids[i]
    row_features = X_test_df.iloc[idx].to_dict()
    actual_label = int(y_test_arr[idx])
    m1_prob = float(baseline_metrics["y_prob"][idx])
    m1_pred = int(baseline_metrics["y_pred"][idx])
    m2_prob = float(trustworthy_metrics["y_prob"][idx])
    m2_pred = int(trustworthy_metrics["y_pred"][idx])
    
    case_data = {
        "case_id": case_id,
        "actual_label": actual_label,
        "model1_pred": m1_pred,
        "model1_prob": round(m1_prob, 4),
        "model1_correct": (m1_pred == actual_label),
        "model2_pred": m2_pred,
        "model2_prob": round(m2_prob, 4),
        "model2_correct": (m2_pred == actual_label),
    }
    case_data.update(row_features)
    sample_cases_list.append(case_data)

sample_cases_df = pd.DataFrame(sample_cases_list)

# Local SHAP for the 15 cases
sample_records_df = X_test_df.iloc[selected_indices]
local_shap_values, base_value = explainability.compute_local_shap(
    trustworthy_model,
    sample_records_df,
    feature_order,
    X_train_df,
    n_background=200
)
print(f"    Local SHAP values computed for all 15 sample cases.")

# 6.3 Fairness Audit (Sex, Age, Income)
print("  > Auditing demographic fairness across Sex, Age, and Income...")
fairness_dict = fairness.compute_fairness_both_models(
    baseline_preds=baseline_metrics["y_pred"],
    trustworthy_preds=trustworthy_metrics["y_pred"],
    y_true=y_test,
    y_prob_base=baseline_metrics["y_prob"],
    y_prob_trust=trustworthy_metrics["y_prob"],
    X_test_df=X_test_df
)
print("    Demographic Parity Differences:")
for attr, s in fairness_dict["comparison_summary"].items():
    print(f"      - {attr}: Baseline={s['demographic_parity_diff_baseline']:.4f}, Trustworthy={s['demographic_parity_diff_trustworthy']:.4f}")

# 6.4 Calibration & Reliability
print("  > Calculating Reliability Curves and Expected Calibration Error (ECE)...")
calibration_dict = calibration.compute_calibration_both_models(
    y_true=y_test,
    y_prob_baseline=baseline_metrics["y_prob"],
    y_prob_trustworthy=trustworthy_metrics["y_prob"],
    n_bins=10
)
print(f"    Baseline ECE:    {calibration_dict['baseline_calibration']['ece']:.4f} (Brier: {baseline_metrics['brier_score']:.4f})")
print(f"    Trustworthy ECE: {calibration_dict['trustworthy_calibration']['ece']:.4f} (Brier: {trustworthy_metrics['brier_score']:.4f})")

# 6.5 Controlled Robustness Testing
print("  > Running Controlled Perturbation Sensitivity Tests (500 samples x 6 perturbations)...")
robustness_results_df = robustness.run_all_perturbations(
    baseline_model=baseline_model,
    trustworthy_model=trustworthy_model,
    X_test_df=X_test_df,
    y_test=y_test,
    n_samples=500,
    random_state=RANDOM_SEED
)
robustness_summary_df = robustness.compute_robustness_summary(robustness_results_df)
print(f"    Robustness evaluations completed: {len(robustness_results_df):,} evaluations logged.")

# 7. EXPERIMENTAL FAIRNESS MITIGATION
print("\n[7/8] Running Experimental Post-Processing Fairness Mitigation on Income...")
mitigation_results = mitigation.run_mitigation_experiment(
    baseline_model=baseline_model,
    X_val=X_val,
    y_val=y_val,
    X_val_df=X_val_df,
    X_test=X_test,
    y_test=y_test,
    X_test_df=X_test_df,
    sensitive_col="Income",
    random_state=RANDOM_SEED
)
print(f"    Income Equalized Odds Diff: {mitigation_results['baseline']['equalized_odds_difference']:.4f} -> {mitigation_results['mitigated']['equalized_odds_difference']:.4f}")
print(f"    Accuracy Impact: {mitigation_results['baseline']['accuracy']:.4f} -> {mitigation_results['mitigated']['accuracy']:.4f}")

# 8. SAVE ALL 11 ARTIFACTS
print("\n[8/8] Saving all 11 standardized artifacts to artifacts/ directory...")
artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# 1. Models
artifact_io.save_model(baseline_model, "baseline_model.joblib", artifacts_dir=artifacts_dir)
artifact_io.save_model(trustworthy_model, "trustworthy_model.joblib", artifacts_dir=artifacts_dir)

# 2. Metadata & Metrics
extra_meta = {
    "deduplication_sensitivity_experiment": dedup_sensitivity,
    "fairness_mitigation_experiment": {
        "sensitive_attribute": "Income",
        "method": "ThresholdOptimizer (Equalized Odds)",
        "accuracy_before": mitigation_results["baseline"]["accuracy"],
        "accuracy_after": mitigation_results["mitigated"]["accuracy"],
        "equalized_odds_diff_before": mitigation_results["baseline"]["equalized_odds_difference"],
        "equalized_odds_diff_after": mitigation_results["mitigated"]["equalized_odds_difference"],
        "operational_requirement": "Group-specific decision thresholds require sensitive attribute (Income) at inference time."
    }
}
artifact_io.save_model_metadata(baseline_metrics, trustworthy_metrics, feature_names, artifacts_dir=artifacts_dir, extra_metadata=extra_meta)
artifact_io.save_evaluation_metrics(baseline_metrics, trustworthy_metrics, artifacts_dir=artifacts_dir)

# 3. Fairness
fairness.save_fairness_artifacts(fairness_dict, artifacts_dir=artifacts_dir)

# 4. Calibration
calibration.save_calibration_artifacts(calibration_dict, artifacts_dir=artifacts_dir)

# 5. Robustness
robustness.save_robustness_artifacts(robustness_results_df, robustness_summary_df, artifacts_dir=artifacts_dir)

# 6. Sample Cases
artifact_io.save_sample_cases(sample_cases_df, artifacts_dir=artifacts_dir)

# 7. SHAP
explainability.save_shap_artifacts(global_importance_df, local_shap_values, feature_order, case_ids, artifacts_dir=artifacts_dir)

# FINAL VERIFICATION OF ARTIFACT LOADING
print("\n" + "="*75)
print("ARTIFACT VERIFICATION AUDIT")
print("="*75)
loaded = artifact_io.load_all_artifacts(artifacts_dir=artifacts_dir)
for k, v in loaded.items():
    if v is None:
        print(f"  FAILED: Artifact [{k}] could not be loaded!")
    else:
        info = f"{type(v).__name__}"
        if isinstance(v, pd.DataFrame):
            info += f" ({v.shape[0]}x{v.shape[1]})"
        elif isinstance(v, dict):
            info += f" ({len(v)} keys)"
        print(f"  VERIFIED: [{k:20s}] -> {info}")

all_ok = all(v is not None for v in loaded.values())
if all_ok:
    print("\n>>> SUCCESS: ALL 11 ARTIFACTS GENERATED, SAVED, AND VERIFIED! <<<")
else:
    print("\n>>> ERROR: SOME ARTIFACTS ARE MISSING! <<<")
