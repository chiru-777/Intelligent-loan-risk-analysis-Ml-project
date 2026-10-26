"""
Demo Output Display Script.
Gathers and prints all live outputs from the trained models, evaluation benchmarks,
SHAP explainability engine, What-If simulator, and applicant predictions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure root in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import LoanRiskPredictor, get_sample_applicant
from src.explainability import LoanRiskExplainer
from src.ai_advisor import AICreditAdvisor

def display_all_outputs():
    pred = LoanRiskPredictor()
    
    print("=" * 80)
    print("      INTELLIGENT LOAN RISK ASSESSMENT SYSTEM - COMPLETE SYSTEM OUTPUTS")
    print("          Aditya University - Department of AI & Machine Learning")
    print("=" * 80)

    # 1. Dataset Summary
    print("\n[OUTPUT 1] DATASET OVERVIEW & SUMMARY STATISTICS")
    print("-" * 80)
    df_raw = pd.read_csv(os.path.join(CURRENT_DIR, 'data', 'loan_applicants.csv'))
    print(f"Total Records   : {df_raw.shape[0]:,}")
    print(f"Total Attributes: {df_raw.shape[1]}")
    print("Class Distribution:")
    for cls, cnt in df_raw['risk'].value_counts().items():
        print(f"  • {cls:<12}: {cnt:,} applicants ({cnt/len(df_raw)*100:.1f}%)")
    print("\nSample Records (First 3):")
    cols_to_show = ['age', 'income', 'credit_score', 'loan_amount', 'loan_term', 'debt_to_income_ratio', 'risk']
    print(df_raw[cols_to_show].head(3).to_string(index=False))

    # 2. Model Performance Benchmarks
    print("\n\n[OUTPUT 2] MODEL PERFORMANCE BENCHMARKS (HOLDOUT TEST DATA - 600 SAMPLES)")
    print("-" * 80)
    pct_df = pd.read_csv(os.path.join(CURRENT_DIR, 'outputs', 'model_comparison_percentages.csv'))
    print(pct_df.to_string(index=False))

    # 3. Confusion Matrices
    print("\n\n[OUTPUT 3] DETAILED CONFUSION MATRICES")
    print("-" * 80)
    with open(os.path.join(CURRENT_DIR, 'outputs', 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    print(f"{'Model Architecture':<32} | {'True Neg (Low)':<14} | {'False Pos':<10} | {'False Neg':<10} | {'True Pos (High)':<15}")
    print("-" * 80)
    for k, v in metrics.items():
        cm = v['confusion_matrix']
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        print(f"{v['model_name']:<32} | {tn:<14} | {fp:<10} | {fn:<10} | {tp:<15}")

    # 4. Individual Applicant Prediction
    print("\n\n[OUTPUT 4] MULTI-MODEL RISK PREDICTION (SAMPLE HIGH-RISK APPLICANT)")
    print("-" * 80)
    high_app = get_sample_applicant('high')
    print("Input Applicant Profile:")
    print(f"  Age: {high_app['age']} | Income: ${high_app['income']:,} | Credit Score: {high_app['credit_score']} | Loan: ${high_app['loan_amount']:,}")
    print(f"  Employment: {high_app['employment_status']} | DTI: {high_app['debt_to_income_ratio']:.2f} | Payment History: {high_app['payment_history']}")
    print("\nModel Inferences:")
    res_high = pred.predict_applicant(high_app)
    for k, v in res_high['all_predictions'].items():
        print(f"  • {v['model_name']:<30}: {v['prediction']:<10} (High Risk Probability: {v['risk_percentage']}%)")

    # 4b. AI Advisor Problem Diagnosis & Detailed Matter
    print("\n\n[OUTPUT 5] AI PROBLEM DIAGNOSIS & UNDERWRITING INTELLIGENCE ('MATTER')")
    print("-" * 80)
    advisor = AICreditAdvisor()
    stacking_pred = res_high['all_predictions']['stacking']
    ai_report = advisor.generate_full_diagnostic_report(high_app, stacking_pred)
    print(f"AI Risk Classification: {stacking_pred['prediction']} (Risk Score: {stacking_pred['risk_percentage']}%)")
    print(f"Assigned Risk Tier    : {ai_report['risk_tier']}")
    print(f"Underwriting Action   : {ai_report['action_code']}")
    print("\nIdentified Root-Cause Problems & Metric Deficits:")
    for prob in ai_report['problems']:
        print(f"  • [{prob['severity'].upper()}] {prob['category']}")
        print(f"    Observed: {prob['metric']} | Benchmark: {prob['benchmark']} | Variance: {prob['deficit']}")
        print(f"    Diagnosis: {prob['explanation']}")
    print("\nAI Corrective Action Roadmap:")
    for step in ai_report['action_plan']:
        print(f"  {step}")

    # 6. What-If Simulator Output
    print("\n\n[OUTPUT 6] INTERACTIVE WHAT-IF RISK SIMULATOR")
    print("-" * 80)
    print("Scenario: Applicant restructures debt, raises credit score, and gains employment.")
    mod_app = high_app.copy()
    mod_app.update({
        'credit_score': 740,
        'debt_to_income_ratio': 0.18,
        'employment_status': 'Employed',
        'payment_history': 'On-Time',
        'existing_debts': 5000
    })
    res_mod = pred.predict_applicant(mod_app)
    
    orig_prob = res_high['all_predictions']['stacking']['risk_percentage']
    new_prob = res_mod['all_predictions']['stacking']['risk_percentage']
    delta = new_prob - orig_prob
    
    print(f"  Baseline Classification   : {res_high['all_predictions']['stacking']['prediction']} ({orig_prob}%)")
    print(f"  Simulated Classification  : {res_mod['all_predictions']['stacking']['prediction']} ({new_prob}%)")
    print(f"  Risk Probability Shift    : {delta:+.2f}%")
    print("  Parameter Modifications:")
    print(f"    - Credit Score      : {high_app['credit_score']} -> {mod_app['credit_score']}")
    print(f"    - DTI Ratio         : {high_app['debt_to_income_ratio']} -> {mod_app['debt_to_income_ratio']}")
    print(f"    - Employment Status : {high_app['employment_status']} -> {mod_app['employment_status']}")
    print(f"    - Payment History   : {high_app['payment_history']} -> {mod_app['payment_history']}")
    print("  Note: For educational/demonstration purposes only; does not guarantee loan approval.")

    # 7. SHAP Explainability Output
    print("\n\n[OUTPUT 7] EXPLAINABLE AI (SHAP) ATTRIBUTION & NATURAL LANGUAGE")
    print("-" * 80)
    x_vec = pred.preprocessor.transform(pd.DataFrame([high_app]))
    explainer = LoanRiskExplainer(pred.models['gradient_boosting'], pred.preprocessor.selected_feature_names_)
    inst = explainer.explain_instance(x_vec[0], high_app)
    
    print(f"Model Expected Prior (Base Value): {inst['base_value']:.4f}")
    print("Top Feature Marginal Contributions:")
    top_c = inst['contributions'].head(5)
    for idx, row in top_c.iterrows():
        print(f"  {row['feature']:<30}: {row['shap_value']:+.4f} ({row['direction']})")
        
    print("\nPlain-English Risk Driver Narratives:")
    for narrative in inst['narratives'][:4]:
        # Strip markdown symbols for terminal display
        clean_narr = narrative.replace('**', '').replace('🔴 ', '[HIGH RISK DRIVER] ').replace('🟢 ', '[LOW RISK DRIVER] ')
        print(f"  • {clean_narr}")

    print("\n" + "=" * 80)
    print("  LIVE STREAMLIT DASHBOARD ACTIVE AT: http://localhost:8501")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    display_all_outputs()
