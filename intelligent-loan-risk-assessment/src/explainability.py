"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Provides:
- Global feature importance (TreeExplainer)
- Local individual applicant prediction explanation
- Interactive and static visualization generation (Waterfall, Bar, Summary)
- Plain-English natural language interpretation of financial risk drivers
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


class LoanRiskExplainer:
    """
    SHAP-based Explainer for Loan Risk Predictions.
    Default explainer uses Random Forest or Gradient Boosting via TreeExplainer
    for high speed, stability, and exact Shapley values.
    """
    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        
        # Initialize explainer
        # Try TreeExplainer first for tree models
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.is_tree = True
        except Exception:
            # Fallback for non-tree or ensemble models using background sample
            if background_data is not None:
                bg_sample = shap.sample(background_data, min(50, len(background_data)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
            else:
                self.explainer = shap.Explainer(self.model)
            self.is_tree = False

    def explain_global(self, X_sample: np.ndarray, max_display: int = 10) -> pd.DataFrame:
        """
        Calculates global feature importance based on mean absolute SHAP value.
        """
        shap_values = self._compute_shap_values(X_sample)
        # Class 1 (High Risk)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
            
        mean_abs_shap = np.abs(sv).mean(axis=0)
        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_abs_shap
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return df_importance.head(max_display)

    def explain_instance(self, x_vector: np.ndarray, raw_features_dict: Optional[dict] = None) -> Dict[str, Any]:
        """
        Explains an individual prediction for a single applicant.
        Returns:
        - raw SHAP values
        - base_value
        - sorted contributions dataframe
        - plain-English textual explanations
        """
        if len(x_vector.shape) == 1:
            x_2d = x_vector.reshape(1, -1)
        else:
            x_2d = x_vector
            
        shap_values = self._compute_shap_values(x_2d)
        
        # Extract class 1 (High Risk) SHAP values
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]
            
        # Base value (expected value)
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_val = float(expected_val[1]) if len(expected_val) > 1 else float(expected_val[0])
        else:
            base_val = float(expected_val)
            
        contributions = []
        for feat, val in zip(self.feature_names, vals):
            raw_val = None
            if raw_features_dict:
                # Find matching raw feature
                for k, v in raw_features_dict.items():
                    if k in feat.lower():
                        raw_val = v
                        break
                        
            contributions.append({
                'feature': feat,
                'shap_value': float(val),
                'abs_shap': abs(float(val)),
                'direction': 'Increases High-Risk' if val > 0 else 'Reduces High-Risk (Low-Risk Factor)',
                'raw_value': raw_val
            })
            
        contrib_df = pd.DataFrame(contributions).sort_values(by='abs_shap', ascending=False).reset_index(drop=True)
        
        # Generate plain English insights
        narratives = self._generate_narratives(contrib_df, raw_features_dict)
        
        return {
            'base_value': base_val,
            'contributions': contrib_df,
            'narratives': narratives,
            'feature_names': self.feature_names,
            'shap_values': vals
        }

    def _compute_shap_values(self, X: np.ndarray):
        """Helper to compute SHAP values safely."""
        try:
            return self.explainer.shap_values(X)
        except Exception:
            exp = self.explainer(X)
            if hasattr(exp, 'values'):
                return exp.values
            return exp

    def _generate_narratives(self, contrib_df: pd.DataFrame, raw_dict: Optional[dict]) -> List[str]:
        """Translates top SHAP drivers into accessible financial explanations."""
        narratives = []
        top_drivers = contrib_df.head(5)
        
        for _, row in top_drivers.iterrows():
            feat = row['feature']
            shap_val = row['shap_value']
            
            # Simple, plain-English explanations that everyone can understand
            if 'credit_score' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Low Credit Score**: A credit score below standard makes the loan much riskier. It shows past payment trouble.")
                else:
                    narratives.append("🟢 **Good Credit Score**: Having a strong credit score is a major plus. It proves you pay back borrowed money reliably.")
            elif 'debt_to_income' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Too Much Debt (DTI)**: Spending too much of your monthly income on existing debts makes it hard to afford a new loan.")
                else:
                    narratives.append("🟢 **Low Existing Debt (DTI)**: You owe very little compared to your income, so you have plenty of room to make loan payments.")
            elif 'payment_history' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Late or Missed Payments**: Missing payments in the past makes lenders worry you might miss payments on this loan too.")
                else:
                    narratives.append("🟢 **On-Time Payments**: Having a clean record of on-time payments gives lenders strong confidence in you.")
            elif 'income' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Income Level**: Your current income is on the lower side for the requested loan amount.")
                else:
                    narratives.append("🟢 **Healthy Income**: A good, steady income gives you comfortable room to pay your monthly installments.")
            elif 'assets' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Low Savings / Assets**: You have very little savings or property to fall back on if an unexpected emergency happens.")
                else:
                    narratives.append("🟢 **Good Savings / Assets**: Having savings and assets provides a strong safety cushion for the loan.")
            elif 'employment' in feat.lower():
                if shap_val > 0:
                    narratives.append("🔴 **Employment Risk**: Not having steady corporate employment makes your monthly income less certain.")
                else:
                    narratives.append("🟢 **Steady Job**: Having a steady job assures lenders that a regular paycheck is coming in every month.")
            else:
                simple_feat = feat.replace('_', ' ').title()
                if shap_val > 0:
                    narratives.append(f"🔴 **{simple_feat}**: This factor pushed your risk higher.")
                else:
                    narratives.append(f"🟢 **{simple_feat}**: This factor helped make your loan safer.")
                
        return narratives

    def save_global_plot(self, X_sample: np.ndarray, output_path: str):
        """Generates and saves a static SHAP summary bar chart."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_imp = self.explain_global(X_sample, max_display=12)
        
        plt.figure(figsize=(8, 5))
        plt.barh(df_imp['Feature'][::-1], df_imp['Importance'][::-1], color='#1f77b4')
        plt.xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=11)
        plt.title('Global Feature Importance (SHAP Analysis)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
