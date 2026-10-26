"""
Complete Training and Evaluation Pipeline Script.
Executes end-to-end:
1. Loads dataset
2. Cleans dataset & removes duplicates
3. Fits Preprocessing pipeline & feature selection on training split (no leakage)
4. Trains SVM, KNN, Random Forest, Gradient Boosting, Soft Voting, and Stacking classifiers
5. Evaluates all models on actual holdout test data
6. Saves model artifacts and preprocessing pipeline to models/
7. Generates and saves evaluation curves, confusion matrices, and SHAP artifacts to outputs/
8. Prints formatted model performance comparison table
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocessing import (
    clean_raw_data,
    LoanDataPreprocessor,
    TARGET_MAPPING,
    TARGET_COL
)
from src.models import (
    get_all_models,
    save_models,
    MODEL_DISPLAY_NAMES
)
from src.evaluation import (
    evaluate_all_models,
    save_evaluation_outputs,
    get_percentage_summary_table
)
from src.explainability import LoanRiskExplainer
from src.database import import_csv_to_sqlite, load_dataset_from_sqlite, DB_PATH


def run_training():
    print("=" * 75)
    print("     INTELLIGENT LOAN RISK ASSESSMENT SYSTEM - MODEL TRAINING PIPELINE     ")
    print("      Aditya University - Department of AI & Machine Learning             ")
    print("=" * 75)
    
    # 1. Load Dataset
    data_path = os.path.join(CURRENT_DIR, 'data', 'loan_applicants.csv')
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at: {data_path}")
        print("[*] Generating dataset automatically...")
        from data.generate_data import generate_loan_data
        df_gen = generate_loan_data(3000, random_state=42)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df_gen.to_csv(data_path, index=False)
        print(f"[+] Generated dataset saved to {data_path}")

    # Initialize and sync SQLite database
    print(f"[*] Syncing dataset with SQLite database at: {DB_PATH}")
    sqlite_rows = import_csv_to_sqlite()
    print(f"[+] SQLite database connected & synchronized: {sqlite_rows:,} records in 'loan_applicants' table.")

    print(f"[*] Loading dataset from SQLite...")
    raw_df = load_dataset_from_sqlite()
    initial_rows = len(raw_df)
    print(f"[+] Raw dataset loaded from SQLite: {initial_rows} records, {raw_df.shape[1]} columns.")
    
    # 2. Data Cleaning
    print("\n[*] Step 1: Data Cleaning (Removing duplicates, handling string spacing)...")
    cleaned_df = clean_raw_data(raw_df)
    dropped_dups = initial_rows - len(cleaned_df)
    print(f"[+] Duplicates removed: {dropped_dups}. Cleaned dataset rows: {len(cleaned_df)}")
    
    # 3. Train/Test Split
    print("\n[*] Step 2: Stratified Train/Test Split (80% Train, 20% Test)...")
    X = cleaned_df.drop(columns=[TARGET_COL])
    y = cleaned_df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"[+] Training samples: {len(X_train)} | Testing holdout samples: {len(X_test)}")
    
    # 4. Preprocessing & Feature Selection
    print("\n[*] Step 3: Fitting Preprocessor & Feature Selection on Training Set...")
    preprocessor = LoanDataPreprocessor(k_features=14)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Numerical target for metrics & models
    y_train_num = y_train.map(TARGET_MAPPING).to_numpy()
    y_test_num = y_test.map(TARGET_MAPPING).to_numpy()
    
    print(f"[+] Categorical encoding & outlier treatment complete.")
    print(f"[+] Encoded features: {len(preprocessor.encoded_feature_names_)}")
    print(f"[+] Selected features ({len(preprocessor.selected_feature_names_)}): {preprocessor.selected_feature_names_}")
    
    # 5. Model Initialization
    print("\n[*] Step 4: Initializing ML Models & Ensemble Architecture...")
    models = get_all_models(random_state=42)
    trained_models = {}
    
    # 6. Train all models
    print("\n" + "-" * 75)
    print(f"{'Model Architecture':<35} | {'Status':<15} | {'Training Time (s)':<15}")
    print("-" * 75)
    
    for key, model in models.items():
        name = MODEL_DISPLAY_NAMES.get(key, key)
        start_time = time.time()
        model.fit(X_train_processed, y_train_num)
        elapsed = time.time() - start_time
        trained_models[key] = model
        print(f"{name:<35} | {'Completed':<15} | {elapsed:.2f}s")
        
    print("-" * 75)
    
    # 7. Evaluate on actual holdout test dataset
    print("\n[*] Step 5: Evaluating all models on unseen test data...")
    comparison_df, detailed_results = evaluate_all_models(trained_models, X_test_processed, y_test_num)
    
    print("\n" + "=" * 80)
    print("               MODEL EVALUATION PERCENTAGES BENCHMARK (%)                ")
    print("=" * 80)
    pct_summary = get_percentage_summary_table(comparison_df)
    print(pct_summary.to_string(index=False))
    print("=" * 80)
    
    # 8. Save Models & Preprocessor
    models_dir = os.path.join(CURRENT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n[*] Step 6: Saving model artifacts to: {models_dir}")
    save_models(trained_models, models_dir)
    preprocessor_path = os.path.join(models_dir, 'preprocessing.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[+] Preprocessor saved to {preprocessor_path}")
    print(f"[+] All 6 model binaries (.pkl) successfully saved.")
    
    # 9. Save Evaluation Outputs
    outputs_dir = os.path.join(CURRENT_DIR, 'outputs')
    print(f"\n[*] Step 7: Generating and saving evaluation artifacts to: {outputs_dir}")
    save_evaluation_outputs(detailed_results, comparison_df, outputs_dir)
    print(f"[+] Confusion matrices saved in {os.path.join(outputs_dir, 'confusion_matrices')}")
    print(f"[+] ROC curves saved in {os.path.join(outputs_dir, 'roc_curves')}")
    print(f"[+] Model comparison CSV & JSON metrics saved.")
    
    # 10. Generate SHAP Explainer Artifacts
    print("\n[*] Step 8: Generating Explainable AI (SHAP) global summary plot...")
    try:
        explainer = LoanRiskExplainer(
            model=trained_models['gradient_boosting'],
            feature_names=preprocessor.selected_feature_names_
        )
        shap_out_path = os.path.join(outputs_dir, 'feature_importance', 'shap_global_importance.png')
        explainer.save_global_plot(X_test_processed[:150], shap_out_path)
        print(f"[+] SHAP global importance plot saved to: {shap_out_path}")
    except Exception as e:
        print(f"[!] Note on SHAP global plot: {e}")
        
    print("\n" + "=" * 75)
    print(" [+] PIPELINE COMPLETED SUCCESSFULLY! ")
    print(" You can now run the web dashboard using: streamlit run app.py")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    run_training()
