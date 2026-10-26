"""
Inference and Prediction Pipeline Module.
Provides functions and CLI utility to predict loan applicant risk across:
1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Random Forest (RF)
4. Gradient Boosting (GB)
5. Soft Voting Ensemble
6. Stacking Ensemble
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models import load_models, MODEL_DISPLAY_NAMES
from src.preprocessing import INVERSE_TARGET_MAPPING, LoanDataPreprocessor


class LoanRiskPredictor:
    """
    Production-ready Loan Risk Inference Engine.
    Loads trained preprocessor and all 6 models to evaluate any applicant.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(CURRENT_DIR, 'models')
            
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, 'preprocessing.pkl')
        
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}. "
                "Please run 'python train.py' to train and save models."
            )
            
        self.preprocessor: LoanDataPreprocessor = joblib.load(self.preprocessor_path)
        self.models = load_models(models_dir)

    def predict_applicant(self, applicant_data: Union[dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Takes applicant details and returns predictions + risk probabilities
        for all 6 models.
        """
        if isinstance(applicant_data, dict):
            applicant_df = pd.DataFrame([applicant_data])
        elif isinstance(applicant_data, pd.DataFrame):
            applicant_df = applicant_data.copy()
        else:
            raise ValueError("applicant_data must be a dict or a pandas DataFrame")

        # Transform features
        X_vec = self.preprocessor.transform(applicant_df)
        
        predictions = {}
        for key, model in self.models.items():
            pred_num = int(model.predict(X_vec)[0])
            pred_label = INVERSE_TARGET_MAPPING.get(pred_num, 'Unknown')
            
            # Risk probability (Probability of Class 1: High Risk)
            if hasattr(model, 'predict_proba'):
                prob_high_risk = float(model.predict_proba(X_vec)[0][1])
            elif hasattr(model, 'decision_function'):
                df_val = float(model.decision_function(X_vec)[0])
                prob_high_risk = 1.0 / (1.0 + np.exp(-df_val))
            else:
                prob_high_risk = 1.0 if pred_num == 1 else 0.0

            prob_low_risk = 1.0 - prob_high_risk

            predictions[key] = {
                'model_name': MODEL_DISPLAY_NAMES.get(key, key),
                'prediction': pred_label,
                'pred_class': pred_num,
                'prob_high_risk': round(prob_high_risk, 4),
                'prob_low_risk': round(prob_low_risk, 4),
                'risk_percentage': round(prob_high_risk * 100, 2)
            }
            
        # Summary consensus
        high_risk_votes = sum(1 for p in predictions.values() if p['prediction'] == 'High Risk')
        
        return {
            'individual_models': {
                'svm': predictions['svm'],
                'knn': predictions['knn'],
                'random_forest': predictions['random_forest'],
                'gradient_boosting': predictions['gradient_boosting']
            },
            'ensemble_models': {
                'soft_voting': predictions['soft_voting'],
                'stacking': predictions['stacking']
            },
            'all_predictions': predictions,
            'processed_features': X_vec,
            'selected_features': self.preprocessor.selected_feature_names_,
            'high_risk_vote_count': high_risk_votes,
            'total_models': len(predictions)
        }


def get_sample_applicant(risk_type: str = 'low') -> dict:
    """Returns a realistic sample applicant for testing or demonstration."""
    if risk_type.lower() == 'high':
        return {
            'age': 27,
            'income': 26000,
            'credit_score': 540,
            'loan_amount': 28000,
            'employment_status': 'Unemployed',
            'loan_term': 36,
            'existing_debts': 18000,
            'debt_to_income_ratio': 0.68,
            'credit_history': 'Poor',
            'payment_history': 'Missed',
            'assets': 4000,
            'previous_loans': 3
        }
    else:
        return {
            'age': 38,
            'income': 85000,
            'credit_score': 740,
            'loan_amount': 20000,
            'employment_status': 'Employed',
            'loan_term': 36,
            'existing_debts': 8000,
            'debt_to_income_ratio': 0.18,
            'credit_history': 'Good',
            'payment_history': 'On-Time',
            'assets': 95000,
            'previous_loans': 2
        }


if __name__ == '__main__':
    print("=" * 65)
    print("  INTELLIGENT LOAN RISK PREDICTOR - SAMPLE INFERENCE TEST  ")
    print("=" * 65)
    
    predictor = LoanRiskPredictor()
    
    # Test Low Risk Sample
    print("\n[+] Testing Low-Risk Profile Applicant:")
    low_sample = get_sample_applicant('low')
    res_low = predictor.predict_applicant(low_sample)
    for key, p in res_low['all_predictions'].items():
        print(f"  {p['model_name']:<32}: {p['prediction']:<10} (High Risk Prob: {p['risk_percentage']}%)")
        
    # Test High Risk Sample
    print("\n[+] Testing High-Risk Profile Applicant:")
    high_sample = get_sample_applicant('high')
    res_high = predictor.predict_applicant(high_sample)
    for key, p in res_high['all_predictions'].items():
        print(f"  {p['model_name']:<32}: {p['prediction']:<10} (High Risk Prob: {p['risk_percentage']}%)")
        
    print("\n[+] Inference test completed successfully.")
