"""
Intelligent Loan Risk Assessment System - Single Module Web Application
Department of Artificial Intelligence and Machine Learning
Aditya University

Single-Module Unified Flow:
1. Input applicant parameters: Income, Before Loans (Previous Loans), Credit Score, Loan Amount, etc.
2. Click 'Predict Applicant Risk' -> Get immediate 'High Risk' or 'Low Risk' verdict.
3. Below that: AI diagnostic engine analyzes and gives in-depth reasons for the loan risk ('Matter').
4. Integrated What-If Simulator and SHAP explainability inline.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Intelligent Loan Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocessing import (
    clean_raw_data,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    INVERSE_TARGET_MAPPING,
    LoanDataPreprocessor
)
from src.models import (
    load_models,
    MODEL_DISPLAY_NAMES
)
from src.visualization import (
    plot_risk_distribution,
    plot_income_vs_loan,
    plot_credit_score_distribution,
    plot_risk_by_employment,
    plot_shap_waterfall,
    plot_what_if_comparison,
    COLOR_LOW_RISK,
    COLOR_HIGH_RISK
)
from src.explainability import LoanRiskExplainer
from src.ai_advisor import AICreditAdvisor
from predict import LoanRiskPredictor, get_sample_applicant
from src.database import (
    load_dataset_from_sqlite,
    log_prediction_to_sqlite,
    get_database_stats,
    get_prediction_audit_logs,
    DB_PATH
)


# --- CACHED LOADERS ---
@st.cache_data
def load_dataset():
    try:
        return load_dataset_from_sqlite()
    except Exception:
        data_path = os.path.join(CURRENT_DIR, 'data', 'loan_applicants.csv')
        if not os.path.exists(data_path):
            return None
        return pd.read_csv(data_path)


@st.cache_resource
def get_predictor():
    models_dir = os.path.join(CURRENT_DIR, 'models')
    preprocessor_path = os.path.join(models_dir, 'preprocessing.pkl')
    if not os.path.exists(preprocessor_path):
        return None
    try:
        return LoanRiskPredictor(models_dir)
    except Exception:
        return None


@st.cache_data
def load_evaluation_percentages():
    pct_path = os.path.join(CURRENT_DIR, 'outputs', 'model_comparison_percentages.csv')
    if os.path.exists(pct_path):
        return pd.read_csv(pct_path)
    return None


@st.cache_resource
def get_shap_explainer(_model, feature_names):
    return LoanRiskExplainer(model=_model, feature_names=feature_names)


# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #A7F3D0;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FECACA;
    }
    .verdict-banner-low {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }
    .verdict-banner-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR INFORMATION ---
st.sidebar.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/shield-halved.svg", width=50)
st.sidebar.markdown("### **Intelligent Loan Risk**")
st.sidebar.caption("Aditya University\nDept. of AI & Machine Learning")
st.sidebar.markdown("---")

# SQLite Database Status
try:
    db_stats = get_database_stats()
    st.sidebar.success("🟢 **SQLite Connected**")
    st.sidebar.caption(
        f"**Database:** `loan_risk.db` ({db_stats['file_size_kb']} KB)\n\n"
        f"**Records:** {db_stats['applicants_count']:,} applicants\n\n"
        f"**Audit Logs:** {db_stats['audit_logs_count']} logged"
    )
except Exception:
    st.sidebar.warning("⚠️ SQLite: Offline")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🏆 Model Accuracy Benchmarks")
pct_df = load_evaluation_percentages()
if pct_df is not None:
    st.sidebar.dataframe(
        pct_df[['Model', 'Accuracy (%)', 'ROC-AUC (%)']],
        hide_index=True,
        use_container_width=True
    )
else:
    st.sidebar.info("Run `python train.py` to view model benchmark percentages.")

st.sidebar.markdown("---")
st.sidebar.caption("Single-Module Architecture: Fill inputs -> Predict Risk -> AI Reasons & Analysis.")


# --- MAIN HEADER ---
st.markdown('<div class="main-header">🏦 Intelligent Loan Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Single-Module Risk Predictor & Explainable AI Engine — Aditya University</div>', unsafe_allow_html=True)

predictor = get_predictor()
if predictor is None:
    st.error("⚠️ **Trained Models Not Found!**")
    st.info("Please run `python train.py` in your terminal to train and save models to `models/`.")
    st.stop()


# --- PRESET PROFILE BUTTONS ---
st.markdown("##### ⚡ Quick-Fill Sample Profiles")
p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
with p_col1:
    if st.button("🟢 Load Low-Risk Profile", use_container_width=True):
        st.session_state['app_input'] = get_sample_applicant('low')
        st.rerun()
with p_col2:
    if st.button("🔴 Load High-Risk Profile", use_container_width=True):
        st.session_state['app_input'] = get_sample_applicant('high')
        st.rerun()

# Fetch active inputs from session state or default to low
input_data = st.session_state.get('app_input', get_sample_applicant('low'))


# ==============================================================================
# SECTION 1: APPLICANT INFORMATION INPUT FORM
# ==============================================================================
st.markdown("---")
st.subheader("📋 Step 1: Enter Applicant Financial & Personal Information")

with st.form("applicant_risk_form"):
    # Row 1: Key inputs prominently placed as requested (Income, Before Loans, Credit Score)
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    with r1_col1:
        income = st.number_input(
            "💰 Applicant Annual Income ($)",
            min_value=5000,
            max_value=500000,
            value=int(input_data.get('income', 85000)),
            step=1000,
            help="Gross annual earnings of the primary applicant"
        )
    with r1_col2:
        previous_loans = st.number_input(
            "📜 Before Loans (Previous Loans Count)",
            min_value=0,
            max_value=15,
            value=int(input_data.get('previous_loans', 2)),
            step=1,
            help="Total count of prior closed or existing credit facilities"
        )
    with r1_col3:
        credit_score = st.slider(
            "⭐ Credit Score (FICO Score: 300 - 850)",
            min_value=300,
            max_value=850,
            value=int(input_data.get('credit_score', 740)),
            help="Standard institutional FICO score rating"
        )

    # Row 2: Loan details & DTI
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    with r2_col1:
        loan_amount = st.number_input(
            "💵 Requested Loan Amount ($)",
            min_value=1000,
            max_value=300000,
            value=int(input_data.get('loan_amount', 20000)),
            step=1000,
            help="Total principal balance requested"
        )
    with r2_col2:
        loan_term = st.selectbox(
            "⏱️ Loan Term (Months)",
            [12, 24, 36, 48, 60],
            index=[12, 24, 36, 48, 60].index(int(input_data.get('loan_term', 36))),
            help="Amortization duration in months"
        )
    with r2_col3:
        dti = st.slider(
            "⚖️ Debt-to-Income (DTI) Ratio",
            min_value=0.01,
            max_value=1.00,
            value=float(input_data.get('debt_to_income_ratio', 0.18)),
            step=0.01,
            help="Ratio of total monthly debt commitments divided by gross monthly income"
        )

    # Row 3: Employment, Payment History, Credit History
    r3_col1, r3_col2, r3_col3 = st.columns(3)
    with r3_col1:
        emp_options = ['Employed', 'Self-Employed', 'Unemployed']
        emp_curr = input_data.get('employment_status', 'Employed')
        emp_idx = emp_options.index(emp_curr) if emp_curr in emp_options else 0
        employment_status = st.selectbox("👔 Employment Status", emp_options, index=emp_idx)

    with r3_col2:
        pay_options = ['On-Time', 'Delayed', 'Missed']
        pay_curr = input_data.get('payment_history', 'On-Time')
        pay_idx = pay_options.index(pay_curr) if pay_curr in pay_options else 0
        payment_history = st.selectbox("📅 Repayment History", pay_options, index=pay_idx)

    with r3_col3:
        hist_options = ['Poor', 'Fair', 'Good', 'Excellent']
        hist_curr = input_data.get('credit_history', 'Good')
        hist_idx = hist_options.index(hist_curr) if hist_curr in hist_options else 2
        credit_history = st.selectbox("📑 Credit History Rating", hist_options, index=hist_idx)

    # Row 4: Debts, Assets, Age
    r4_col1, r4_col2, r4_col3 = st.columns(3)
    with r4_col1:
        existing_debts = st.number_input(
            "💳 Existing Outstanding Debts ($)",
            min_value=0,
            max_value=300000,
            value=int(input_data.get('existing_debts', 8000)),
            step=500
        )
    with r4_col2:
        assets = st.number_input(
            "🏠 Total Declared Assets ($)",
            min_value=0,
            max_value=1000000,
            value=int(input_data.get('assets', 95000)),
            step=5000
        )
    with r4_col3:
        age = st.number_input(
            "🎂 Applicant Age (Years)",
            min_value=18,
            max_value=85,
            value=int(input_data.get('age', 38)),
            step=1
        )

    submit_predict = st.form_submit_button(
        "🚀 Predict Applicant Risk",
        use_container_width=True,
        type="primary"
    )

if submit_predict:
    # Save submitted values into session state
    st.session_state['app_input'] = {
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'employment_status': employment_status,
        'loan_term': loan_term,
        'existing_debts': existing_debts,
        'debt_to_income_ratio': dti,
        'credit_history': credit_history,
        'payment_history': payment_history,
        'assets': assets,
        'previous_loans': previous_loans
    }
    st.session_state['run_prediction'] = True


# ==============================================================================
# SECTION 2: IMMEDIATE RISK PREDICTION RESULT (HIGH RISK OR LOW RISK)
# ==============================================================================
if st.session_state.get('run_prediction', False):
    applicant_payload = st.session_state['app_input']
    results = predictor.predict_applicant(applicant_payload)
    
    ens = results['ensemble_models']
    stk_pred = ens['stacking']['prediction']
    stk_prob = ens['stacking']['risk_percentage']
    vote_pred = ens['soft_voting']['prediction']
    vote_prob = ens['soft_voting']['risk_percentage']

    # Audit log in SQLite database
    try:
        log_prediction_to_sqlite(
            applicant_data=applicant_payload,
            predicted_risk=stk_pred,
            risk_probability=stk_prob / 100.0,
            model_used="Stacking Ensemble"
        )
    except Exception:
        pass

    st.markdown("---")
    st.subheader("🎯 Step 2: Risk Assessment Prediction Result")

    # Prominent Banner
    if stk_pred == "High Risk":
        banner_html = f"""
        <div class="verdict-banner-high">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.9rem; font-weight: 700; color: #991B1B; text-transform: uppercase;">
                        FINAL CLASSIFICATION VERDICT
                    </span>
                    <h1 style="color: #991B1B; margin: 4px 0 6px 0; font-size: 2.5rem; font-weight: 800;">
                        🚨 HIGH RISK APPLICANT
                    </h1>
                    <p style="margin: 0; color: #7F1D1D; font-size: 1.05rem;">
                        This applicant exhibits a high probability of loan default. Detailed AI root-cause analysis is provided below.
                    </p>
                </div>
                <div style="text-align: right; margin-top: 10px;">
                    <span class="badge-high">HIGH RISK</span>
                    <h2 style="color: #991B1B; margin: 8px 0 0 0; font-weight: 800;">{stk_prob:.1f}% Risk Probability</h2>
                    <span style="color: #7F1D1D; font-size: 0.85rem;">Stacking Ensemble Consensus</span>
                </div>
            </div>
        </div>
        """
    else:
        banner_html = f"""
        <div class="verdict-banner-low">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.9rem; font-weight: 700; color: #065F46; text-transform: uppercase;">
                        FINAL CLASSIFICATION VERDICT
                    </span>
                    <h1 style="color: #065F46; margin: 4px 0 6px 0; font-size: 2.5rem; font-weight: 800;">
                        🛡️ LOW RISK APPLICANT
                    </h1>
                    <p style="margin: 0; color: #064E3B; font-size: 1.05rem;">
                        Applicant meets creditworthiness standards with healthy debt capacity and verified repayment behavior.
                    </p>
                </div>
                <div style="text-align: right; margin-top: 10px;">
                    <span class="badge-low">LOW RISK</span>
                    <h2 style="color: #065F46; margin: 8px 0 0 0; font-weight: 800;">{stk_prob:.1f}% Risk Probability</h2>
                    <span style="color: #064E3B; font-size: 0.85rem;">Stacking Ensemble Consensus</span>
                </div>
            </div>
        </div>
        """
    st.markdown(banner_html, unsafe_allow_html=True)

    # Multi-Model Consensus Cards
    st.markdown("##### 🔍 Multi-Algorithm Predictions & Agreement")
    ind = results['individual_models']
    col_stk, col_vot, col_gb, col_rf, col_svm, col_knn = st.columns(6)

    models_display = [
        (col_stk, "Stacking Ensemble", stk_pred, stk_prob, "96.67% Acc"),
        (col_vot, "Soft Voting", vote_pred, vote_prob, "96.50% Acc"),
        (col_gb, "Gradient Boosting", ind['gradient_boosting']['prediction'], ind['gradient_boosting']['risk_percentage'], "96.17% Acc"),
        (col_rf, "Random Forest", ind['random_forest']['prediction'], ind['random_forest']['risk_percentage'], "95.83% Acc"),
        (col_svm, "SVM", ind['svm']['prediction'], ind['svm']['risk_percentage'], "96.00% Acc"),
        (col_knn, "KNN", ind['knn']['prediction'], ind['knn']['risk_percentage'], "94.00% Acc")
    ]

    for col, m_name, m_pred, m_prob, m_acc in models_display:
        with col:
            b_cls = "badge-high" if m_pred == "High Risk" else "badge-low"
            st.markdown(f"""
            <div class="card" style="text-align: center; padding: 0.8rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #4B5563;">{m_name}</div>
                <div style="margin: 6px 0;">
                    <span class="{b_cls}" style="font-size: 0.85rem; padding: 3px 8px;">{m_pred}</span>
                </div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #111827;">{m_prob:.1f}% Risk</div>
                <div style="font-size: 0.75rem; color: #9CA3AF;">Benchmark: {m_acc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ==============================================================================
    # SECTION 3: AI RISK ANALYSIS & REASONS FOR LOAN RISK ("MATTER")
    # ==============================================================================
    st.markdown("---")
    st.subheader("🤖 Step 3: AI Problem Diagnosis — Reasons for Loan Risk ('Matter')")
    st.caption("Detailed institutional credit intelligence explaining EXACTLY why this applicant is classified as " + stk_pred)

    advisor = AICreditAdvisor()
    ai_dossier = advisor.generate_full_diagnostic_report(
        applicant=applicant_payload,
        prediction_result=ens['stacking'],
        shap_explanation=None
    )

    # Executive Diagnostic Overview
    st.markdown(f"""
    <div class="card" style="border-left: 6px solid {ai_dossier['verdict_color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="metric-title">ASSESSED RISK TIER</span>
                <h3 style="margin: 2px 0; color: #111827;">{ai_dossier['risk_tier']}</h3>
            </div>
            <div>
                <span class="metric-title">RECOMMENDED UNDERWRITING ACTION</span>
                <h4 style="margin: 2px 0; color: {ai_dossier['verdict_color']};">{ai_dossier['action_code']}</h4>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Affordability & Stress Metrics Bar
    fm = ai_dossier['financial_metrics']
    fm1, fm2, fm3, fm4 = st.columns(4)
    with fm1:
        st.metric("Estimated Monthly EMI", f"${fm['estimated_monthly_emi']:,.2f}", help="Standard monthly repayment installment")
    with fm2:
        st.metric("Monthly EMI Burden", f"{fm['monthly_burden_pct']:.1f}%", help="Share of gross monthly salary committed to EMI")
    with fm3:
        st.metric("Loan-to-Income Ratio", f"{fm['loan_to_income_pct']:.1f}%", help="Requested principal divided by annual income")
    with fm4:
        st.metric("Asset Collateral Coverage", f"{fm['asset_coverage_pct']:.1f}%", help="Total assets divided by loan amount")

    # Detailed Root Cause Reasons & Matter in Simple Words
    if stk_pred == "High Risk":
        st.markdown("#### 🚨 Exact Reasons Why This Loan is High Risk (In Simple Words):")
        st.markdown("Below is a breakdown of every problem the AI found in the applicant's profile:")
        if ai_dossier['problems']:
            for idx, p in enumerate(ai_dossier['problems'], 1):
                sev = p['severity']
                sev_color = "#DC2626" if sev == "Critical" else ("#EA580C" if sev == "High" else "#D97706")
                st.markdown(f"""
                <div class="card" style="border-left: 5px solid {sev_color}; margin-bottom: 0.9rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #111827;">❌ Reason #{idx}: {p['category']}</h4>
                        <span style="background: {sev_color}22; color: {sev_color}; padding: 3px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem;">
                            {sev.upper()} PROBLEM
                        </span>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.92rem; color: #374151;">
                        • <b>Applicant's Value:</b> <code>{p['metric']}</code> &nbsp;|&nbsp;
                        • <b>Normal Safe Level:</b> <code>{p['benchmark']}</code> &nbsp;|&nbsp;
                        • <b>Difference:</b> <span style="color: {sev_color}; font-weight: 700;">{p['deficit']}</span>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 0.95rem; color: #374151; line-height: 1.5; background: #F9FAFB; padding: 10px; border-radius: 6px;">
                        💡 <b>Simple Explanation:</b> {p['explanation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No single parameter is in severe distress, but having multiple moderate debts together elevates default risk.")

        # Prescriptive Action Plan in Simple Words
        st.markdown("#### 🛠️ AI Advice: How to Fix These Problems and Get Approved (Simple Steps):")
        st.markdown("Here is what the applicant can do to turn this loan into an approved Low-Risk loan:")
        for step in ai_dossier['action_plan']:
            st.markdown(f"- {step}")

    else:
        st.markdown("#### 🛡️ Exact Reasons Why This Loan is Low Risk (In Simple Words):")
        st.markdown("The AI found that the applicant has strong financial habits and meets all safety criteria:")
        for idx, s in enumerate(ai_dossier['strengths'], 1):
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #10B981; margin-bottom: 0.8rem;">
                <h4 style="margin: 0; color: #065F46;">✅ Reason #{idx}: {s['category']}</h4>
                <div style="margin-top: 5px; font-size: 0.92rem; color: #374151;">
                    • <b>Applicant's Value:</b> <code>{s['metric']}</code>
                </div>
                <p style="margin: 8px 0 0 0; font-size: 0.95rem; color: #064E3B; background: #ECFDF5; padding: 8px 10px; border-radius: 6px;">
                    💡 <b>Why this is great:</b> {s['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Explainable AI (SHAP) Waterfall Chart Inline
    st.markdown("#### 🧠 Explainable AI: Feature Contribution Waterfall (SHAP)")
    x_vec = predictor.preprocessor.transform(pd.DataFrame([applicant_payload]))
    explainer = get_shap_explainer(
        _model=predictor.models['gradient_boosting'],
        feature_names=predictor.preprocessor.selected_feature_names_
    )
    with st.spinner("Computing Shapley values..."):
        instance_exp = explainer.explain_instance(x_vec[0], applicant_payload)

    sh_col1, sh_col2 = st.columns([1.3, 1])
    with sh_col1:
        st.plotly_chart(
            plot_shap_waterfall(
                instance_exp['contributions'],
                instance_exp['base_value'],
                max_display=8
            ),
            use_container_width=True
        )
    with sh_col2:
        st.markdown("##### 💬 Plain-English Risk Drivers")
        for narrative in instance_exp['narratives'][:4]:
            st.markdown(f"- {narrative}")
        st.caption(f"Base Expected Prior Value: `{instance_exp['base_value']:.3f}`")

    # ==============================================================================
    # EXPANDERS: WHAT-IF SIMULATOR & FORMAL MEMORANDUM (IN THE SAME MODULE)
    # ==============================================================================
    st.markdown("---")
    with st.expander("🔄 Interactive What-If Risk Simulator (Test Changes Live)", expanded=False):
        st.markdown("##### Explore how changes to Credit Score, Income, or Debt reduce risk in real time:")
        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            sim_income = st.slider("Simulate Income ($)", 10000, 250000, int(applicant_payload['income']), step=2000)
            sim_credit = st.slider("Simulate Credit Score", 300, 850, int(applicant_payload['credit_score']), step=5)
        with w_c2:
            sim_loan = st.slider("Simulate Loan Amount ($)", 2000, 100000, int(applicant_payload['loan_amount']), step=1000)
            sim_dti = st.slider("Simulate DTI Ratio", 0.05, 0.95, float(applicant_payload['debt_to_income_ratio']), step=0.01)
        with w_c3:
            sim_emp = st.selectbox("Simulate Employment", ['Employed', 'Self-Employed', 'Unemployed'], index=['Employed', 'Self-Employed', 'Unemployed'].index(applicant_payload['employment_status']))
            sim_pay = st.selectbox("Simulate Payment History", ['On-Time', 'Delayed', 'Missed'], index=['On-Time', 'Delayed', 'Missed'].index(applicant_payload['payment_history']))

        sim_payload = applicant_payload.copy()
        sim_payload.update({
            'income': sim_income,
            'credit_score': sim_credit,
            'loan_amount': sim_loan,
            'debt_to_income_ratio': sim_dti,
            'employment_status': sim_emp,
            'payment_history': sim_pay
        })

        sim_res = predictor.predict_applicant(sim_payload)['all_predictions']['stacking']
        sim_orig_prob = stk_prob / 100.0
        sim_new_prob = sim_res['prob_high_risk']
        sim_shift = (sim_new_prob - sim_orig_prob) * 100.0

        st.plotly_chart(
            plot_what_if_comparison(
                prob_original=sim_orig_prob,
                prob_new=sim_new_prob,
                orig_label=stk_pred,
                new_label=sim_res['prediction']
            ),
            use_container_width=True
        )
        st.caption(f"Simulated Shift: **{sim_shift:+.1f}%** (From {stk_pred} to {sim_res['prediction']})")

    with st.expander("📜 View Formal AI Underwriting Memorandum ('Matter Dossier')", expanded=False):
        st.text_area("Full Credit Matter Dossier:", value=ai_dossier['matter_report'], height=300)
        st.download_button(
            label="📥 Download Formal Credit Memorandum ('Matter')",
            data=ai_dossier['matter_report'],
            file_name=f"Credit_Assessment_Matter_{applicant_payload.get('credit_score', 600)}_{stk_pred.replace(' ', '_')}.txt",
            mime="text/plain"
        )
