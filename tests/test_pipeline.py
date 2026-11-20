# Integration test suite for Trustworthy Diabetes Risk-Screening
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    artifact_io,
)

def run_all_tests():
    print("=== 1. DATA LOADER CONTRACT & VALIDATION ===")
    feats = data_loader.get_feature_names()
    assert len(feats) == 21
    groups = data_loader.get_feature_groups()
    assert len(groups["continuous"]) == 3
    assert len(groups["ordinal"]) == 4
    assert len(groups["binary"]) == 14
    data_dict = data_loader.get_data_dictionary()
    assert len(data_dict) == 21

    csv_p = os.path.join(PROJECT_ROOT, "data", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    raw_df = data_loader.load_and_validate_dataset(csv_p)
    clean_df = data_loader.remove_duplicates(raw_df)
    assert len(clean_df) == 69057

    print("=== 2. DATA SPLITTING (TEST ON SLICE FOR SPEED) ===")
    slice_df = clean_df.sample(n=1200, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = data_loader.split_dataset(slice_df, random_state=42)
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_val_df = pd.DataFrame(X_val, columns=feat_names)
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    print("=== 3. MODEL 1 (BASELINE) TRAINING & EVALUATION ===")
    m1 = model_baseline.train_baseline_model(X_train, y_train, random_state=42)
    eval1 = model_baseline.evaluate_model(m1, X_test, y_test, threshold=0.5)
    print(f"Model 1 Accuracy: {eval1['accuracy']:.4f}, AUC: {eval1['roc_auc']:.4f}, Brier: {eval1['brier_score']:.4f}")

    print("=== 4. MODEL 2 (TRUST-AWARE WITH FROZENESTIMATOR) TRAINING & EVALUATION ===")
    m2 = model_trustworthy.train_trustworthy_model(X_train, y_train, X_val, y_val, random_state=42)
    eval2 = model_trustworthy.evaluate_trustworthy_model(m2, X_test, y_test, threshold=0.5)
    print(f"Model 2 Accuracy: {eval2['accuracy']:.4f}, AUC: {eval2['roc_auc']:.4f}, Brier: {eval2['brier_score']:.4f}")

    print("=== 5. SHAP EXPLAINABILITY ===")
    out_features = preprocessing.get_feature_order_after_transform()
    shap_vals, exp_feats = explainability.compute_global_shap(m2, X_train_df, out_features, n_background=30, n_explain=40, random_state=42)
    imp_df = explainability.compute_global_importance(shap_vals, exp_feats)
    print("Top SHAP features:", imp_df["feature"].head(3).tolist())

    loc_shap, base_val = explainability.compute_local_shap(m2, X_test_df.iloc[:5], exp_feats, X_train_df, n_background=30)
    assert loc_shap.shape == (5, 21)

    print("=== 6. FAIRNESS AUDIT ===")
    fair_dict = fairness.compute_fairness_both_models(eval1["y_pred"], eval2["y_pred"], y_test, eval1["y_prob"], eval2["y_prob"], X_test_df)
    assert "baseline_model" in fair_dict and "trustworthy_model" in fair_dict

    print("=== 7. CALIBRATION EVALUATION ===")
    cal_dict = calibration.compute_calibration_both_models(y_test, eval1["y_prob"], eval2["y_prob"], n_bins=5)
    print(f"Model 1 ECE: {cal_dict['baseline_calibration']['ece']:.4f}, Model 2 ECE: {cal_dict['trustworthy_calibration']['ece']:.4f}")

    print("=== 8. ROBUSTNESS SENSITIVITY TESTING ===")
    rob_df = robustness.run_all_perturbations(m1, m2, X_test_df, y_test, n_samples=30, random_state=42)
    rob_sum = robustness.compute_robustness_summary(rob_df)
    assert len(rob_sum) > 0

    print("=== 9. MITIGATION EXPERIMENT (FITTED ON VAL, EVALUATED ON TEST) ===")
    mit_res = mitigation.run_mitigation_experiment(
        m1, X_val, y_val, X_val_df, X_test, y_test, X_test_df, sensitive_col="Income", random_state=42
    )
    rep = mitigation.format_mitigation_report(mit_res)
    assert len(rep) > 50
    print("Mitigation report verified")

    print("=== 10. ARTIFACT IO TESTS ===")
    test_art_dir = os.path.join(PROJECT_ROOT, "artifacts")
    os.makedirs(test_art_dir, exist_ok=True)
    m1_path = artifact_io.save_model(m1, "test_baseline.joblib")
    loaded_m1 = artifact_io.load_model("test_baseline.joblib")
    assert loaded_m1 is not None
    if os.path.exists(m1_path):
        os.remove(m1_path)

    print("\n" + "="*60)
    print("ALL 10 MODULES PASSED COMPLETE INTEGRATION CHECKS!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
