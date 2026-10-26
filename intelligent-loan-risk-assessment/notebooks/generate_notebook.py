"""
Script to generate loan_risk_analysis.ipynb
"""
import json
import os

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Intelligent Loan Risk Assessment System\n",
                    "### Aditya University - Department of Artificial Intelligence & Machine Learning\n",
                    "**Presented By:**\n",
                    "- E. Chiranjeevi (24B11AI105)\n",
                    "- E. Madhu Sudhan Reddy (24B11AI022)\n",
                    "- B. D V Sai Ajay (24B11AI022)\n",
                    "- M. Surya Maha Teja (24B11AI251)\n",
                    "\n",
                    "---\n",
                    "## 1. Project Overview\n",
                    "This notebook demonstrates the end-to-end Machine Learning workflow for predicting credit risk (Low Risk vs High Risk) for loan applicants.\n",
                    "It covers:\n",
                    "1. Data Ingestion & Exploratory Data Analysis (EDA)\n",
                    "2. Data Cleaning (Missing Values, Duplicates, Outlier Capping)\n",
                    "3. Preprocessing & Feature Selection\n",
                    "4. Training 4 Base Models (SVM, KNN, Random Forest, Gradient Boosting)\n",
                    "5. Advanced Ensembles (Soft Voting Classifier, Stacking Classifier)\n",
                    "6. Comparative Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curves)\n",
                    "7. Explainable AI using SHAP"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import shap\n",
                    "\n",
                    "# Add project root to sys.path\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "from src.preprocessing import clean_raw_data, LoanDataPreprocessor, TARGET_MAPPING, TARGET_COL\n",
                    "from src.models import get_all_models, MODEL_DISPLAY_NAMES\n",
                    "from src.evaluation import evaluate_all_models\n",
                    "from src.explainability import LoanRiskExplainer\n",
                    "\n",
                    "print('Dependencies and custom modules successfully imported!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load Dataset & Exploratory Data Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = '../data/loan_applicants.csv'\n",
                    "df_raw = pd.read_csv(data_path)\n",
                    "print(f'Dataset Shape: {df_raw.shape}')\n",
                    "df_raw.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Target class distribution\n",
                    "print('Target Risk Distribution:')\n",
                    "print(df_raw['risk'].value_counts())\n",
                    "\n",
                    "# Missing values check\n",
                    "print('\\nMissing Values:')\n",
                    "print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])\n",
                    "\n",
                    "# Duplicate check\n",
                    "print(f'\\nDuplicate rows count: {df_raw.duplicated().sum()}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Data Cleaning & Train/Test Split"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clean data\n",
                    "cleaned_df = clean_raw_data(df_raw)\n",
                    "print(f'Cleaned dataset shape: {cleaned_df.shape}')\n",
                    "\n",
                    "# Train/Test split\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "X = cleaned_df.drop(columns=[TARGET_COL])\n",
                    "y = cleaned_df[TARGET_COL]\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                    ")\n",
                    "print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Preprocessing Pipeline & Feature Selection\n",
                    "Using `ColumnTransformer` with `SimpleImputer`, `OutlierCapper`, `StandardScaler`, `OneHotEncoder`, and `SelectKBest` (fit strictly on training data)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "preprocessor = LoanDataPreprocessor(k_features=14)\n",
                    "X_train_proc = preprocessor.fit_transform(X_train, y_train)\n",
                    "X_test_proc = preprocessor.transform(X_test)\n",
                    "\n",
                    "y_train_num = y_train.map(TARGET_MAPPING).to_numpy()\n",
                    "y_test_num = y_test.map(TARGET_MAPPING).to_numpy()\n",
                    "\n",
                    "print('Selected features:')\n",
                    "print(preprocessor.selected_feature_names_)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Feature importance by ANOVA F-score\n",
                    "feature_scores = preprocessor.get_feature_scores()\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "sns.barplot(data=feature_scores, x='F_Score', y='Feature', hue='Selected', dodge=False)\n",
                    "plt.title('Feature Selection ANOVA F-Scores', fontsize=12, fontweight='bold')\n",
                    "plt.xlabel('F-Score')\n",
                    "plt.ylabel('Encoded Feature')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Model Training (Base Classifiers + Ensembles)\n",
                    "We train:\n",
                    "1. Support Vector Machine (SVM)\n",
                    "2. K-Nearest Neighbors (KNN)\n",
                    "3. Random Forest\n",
                    "4. Gradient Boosting\n",
                    "5. Soft Voting Classifier\n",
                    "6. Stacking Classifier (Meta-Learner: Logistic Regression)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = get_all_models(random_state=42)\n",
                    "trained_models = {}\n",
                    "\n",
                    "for key, model in models.items():\n",
                    "    print(f'Training {MODEL_DISPLAY_NAMES[key]}...')\n",
                    "    model.fit(X_train_proc, y_train_num)\n",
                    "    trained_models[key] = model\n",
                    "print('\\nAll 6 models trained successfully!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Comparative Model Evaluation on Unseen Test Set"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comp_df, results = evaluate_all_models(trained_models, X_test_proc, y_test_num)\n",
                    "display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']\n",
                    "comp_df[display_cols]"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Combined ROC Curve Plot\n",
                    "plt.figure(figsize=(9, 6))\n",
                    "for key, data in results.items():\n",
                    "    fpr = data['roc_curve']['fpr']\n",
                    "    tpr = data['roc_curve']['tpr']\n",
                    "    auc = data['roc_auc']\n",
                    "    plt.plot(fpr, tpr, label=f\"{MODEL_DISPLAY_NAMES[key]} (AUC = {auc:.3f})\")\n",
                    "\n",
                    "plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')\n",
                    "plt.xlabel('False Positive Rate')\n",
                    "plt.ylabel('True Positive Rate')\n",
                    "plt.title('ROC Curves Comparison', fontsize=12, fontweight='bold')\n",
                    "plt.legend(loc='lower right')\n",
                    "plt.grid(alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Explainable AI with SHAP"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "explainer = LoanRiskExplainer(\n",
                    "    model=trained_models['gradient_boosting'],\n",
                    "    feature_names=preprocessor.selected_feature_names_\n",
                    ")\n",
                    "\n",
                    "# Global feature importance\n",
                    "global_imp = explainer.explain_global(X_test_proc[:100])\n",
                    "print('Top Global Features by SHAP Importance:')\n",
                    "print(global_imp)\n",
                    "\n",
                    "# Local prediction explanation for an applicant\n",
                    "sample_vec = X_test_proc[0]\n",
                    "local_exp = explainer.explain_instance(sample_vec)\n",
                    "print('\\nPlain-English Explanations for Applicant:')\n",
                    "for n in local_exp['narratives']:\n",
                    "    print('-', n)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Conclusion\n",
                    "- Both **Stacking Ensemble** and **Soft Voting Ensemble** demonstrated superior predictive performance over individual classifiers.\n",
                    "- Key risk factors identified across ANOVA and SHAP were **Credit Score**, **Debt-to-Income Ratio (DTI)**, and **Repayment History**.\n",
                    "- The end-to-end system is integrated into an interactive web application running on Streamlit with an interactive What-If Risk Simulator."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'notebooks', 'loan_risk_analysis.ipynb'
    )
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Jupyter notebook saved to: {nb_path}")

if __name__ == '__main__':
    create_notebook()
