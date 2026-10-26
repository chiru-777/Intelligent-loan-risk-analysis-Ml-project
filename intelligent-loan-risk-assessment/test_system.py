"""
End-to-End System Verification Test.
Tests data loading, preprocessing, model loading, inference across all 6 models,
SHAP explainability generation, What-If simulation calculations, and output artifact existence.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocessing import clean_raw_data, TARGET_COL
from predict import LoanRiskPredictor, get_sample_applicant
from src.explainability import LoanRiskExplainer


def run_system_verification():
    print("=" * 70)
    print("  RUNNING COMPREHENSIVE SYSTEM VERIFICATION TESTS")
    print("=" * 70)

    # 1. Dataset & SQLite Check
    data_path = os.path.join(CURRENT_DIR, 'data', 'loan_applicants.csv')
    assert os.path.exists(data_path), "Dataset file missing!"
    df = pd.read_csv(data_path)
    assert len(df) > 1000, f"Dataset too small: {len(df)} rows"
    print(f"[PASS] CSV Dataset verified: {len(df)} rows, {df.shape[1]} columns.")

    from src.database import load_dataset_from_sqlite, get_database_stats, log_prediction_to_sqlite
    db_df = load_dataset_from_sqlite()
    assert len(db_df) > 1000, "SQLite dataset empty or failed to load!"
    db_stats = get_database_stats()
    assert db_stats['applicants_count'] > 1000, "SQLite applicants count mismatch!"
    print(f"[PASS] SQLite Database verified: {db_stats['applicants_count']:,} records in loan_applicants table.")

    # 2. Data Cleaning Check
    df_clean = clean_raw_data(df)
    assert len(df_clean) <= len(df), "Cleaning failed to remove duplicates"
    print(f"[PASS] Data cleaning verified: {len(df_clean)} unique rows.")

    # 3. Model & Preprocessor Loading Check
    predictor = LoanRiskPredictor()
    assert len(predictor.models) == 6, f"Expected 6 models, found {len(predictor.models)}"
    expected_models = ['svm', 'knn', 'random_forest', 'gradient_boosting', 'soft_voting', 'stacking']
    for m in expected_models:
        assert m in predictor.models, f"Missing model: {m}"
    print(f"[PASS] Preprocessor and all 6 models loaded successfully: {list(predictor.models.keys())}")

    # 4. Feature Selection Check
    selected_features = predictor.preprocessor.selected_feature_names_
    assert len(selected_features) > 0, "No features were selected!"
    print(f"[PASS] Feature selection verified: {len(selected_features)} features selected: {selected_features}")

    # 5. Inference Check (Low-Risk Profile)
    low_profile = get_sample_applicant('low')
    res_low = predictor.predict_applicant(low_profile)
    assert res_low['ensemble_models']['stacking']['prediction'] == 'Low Risk', "Low-risk applicant misclassified by stacking"
    assert res_low['ensemble_models']['soft_voting']['prediction'] == 'Low Risk', "Low-risk applicant misclassified by voting"
    print(f"[PASS] Low-risk inference verified: Stacking -> {res_low['ensemble_models']['stacking']['risk_percentage']}%")

    # 6. Inference Check (High-Risk Profile)
    high_profile = get_sample_applicant('high')
    res_high = predictor.predict_applicant(high_profile)
    assert res_high['ensemble_models']['stacking']['prediction'] == 'High Risk', "High-risk applicant misclassified by stacking"
    assert res_high['ensemble_models']['soft_voting']['prediction'] == 'High Risk', "High-risk applicant misclassified by voting"
    print(f"[PASS] High-risk inference verified: Stacking -> {res_high['ensemble_models']['stacking']['risk_percentage']}%")

    # 7. What-If Simulator Scenario Check
    modified_high = high_profile.copy()
    # Drastically improve credit score and reduce DTI
    modified_high['credit_score'] = 780
    modified_high['debt_to_income_ratio'] = 0.15
    modified_high['payment_history'] = 'On-Time'
    modified_high['employment_status'] = 'Employed'
    res_modified = predictor.predict_applicant(modified_high)
    orig_prob = res_high['ensemble_models']['stacking']['prob_high_risk']
    new_prob = res_modified['ensemble_models']['stacking']['prob_high_risk']
    prob_delta = new_prob - orig_prob
    assert prob_delta < 0, f"What-If simulation failed: probability did not decrease (delta={prob_delta})"
    print(f"[PASS] What-If simulator verified: Original={orig_prob*100:.1f}%, Simulated={new_prob*100:.1f}%, Shift={prob_delta*100:.1f}%")

    # 8. SHAP Explainability Check
    explainer = LoanRiskExplainer(
        model=predictor.models['gradient_boosting'],
        feature_names=selected_features
    )
    x_vec = predictor.preprocessor.transform(pd.DataFrame([high_profile]))
    exp = explainer.explain_instance(x_vec[0], high_profile)
    assert len(exp['contributions']) == len(selected_features), "Mismatch in SHAP contribution count"
    assert len(exp['narratives']) > 0, "No natural language narratives generated"
    print(f"[PASS] SHAP explainability verified: Generated {len(exp['narratives'])} natural language risk narratives.")

    # 9. Artifact Files Check
    required_artifacts = [
        os.path.join(CURRENT_DIR, 'outputs', 'metrics.json'),
        os.path.join(CURRENT_DIR, 'outputs', 'model_comparison.csv'),
        os.path.join(CURRENT_DIR, 'outputs', 'roc_curves', 'combined_roc_curve.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'feature_importance', 'shap_global_importance.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'stacking_confusion_matrix.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'soft_voting_confusion_matrix.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'gradient_boosting_confusion_matrix.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'random_forest_confusion_matrix.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'svm_confusion_matrix.png'),
        os.path.join(CURRENT_DIR, 'outputs', 'confusion_matrices', 'knn_confusion_matrix.png')
    ]
    for path in required_artifacts:
        assert os.path.exists(path), f"Required artifact missing: {path}"
    print(f"[PASS] All {len(required_artifacts)} evaluation artifact files verified on disk.")

    # 10. Metrics Validation
    with open(os.path.join(CURRENT_DIR, 'outputs', 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    for model_key, m_data in metrics.items():
        assert m_data['accuracy'] > 0.85, f"Suspiciously low accuracy for {model_key}: {m_data['accuracy']}"
        assert m_data['roc_auc'] > 0.90, f"Suspiciously low ROC-AUC for {model_key}: {m_data['roc_auc']}"
    print("[PASS] Evaluation metrics verified: All models exceed 85% accuracy and 90% ROC-AUC.")

    print("\n" + "=" * 70)
    print("  ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 70)

if __name__ == '__main__':
    run_system_verification()
