"""
Model Evaluation and Metrics Module.
Calculates:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix
- ROC Curve data (FPR, TPR, thresholds)
Generates and saves visual evaluation artifacts (Confusion Matrices, ROC curves).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from src.models import MODEL_DISPLAY_NAMES


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a single model on test data.
    Returns metrics dict and raw curves/confusion matrix data.
    """
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        y_proba = (df - df.min()) / (df.max() - df.min() + 1e-8)
    else:
        y_proba = y_pred.astype(float)
        
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        },
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist()
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates all trained models on test dataset.
    Returns a clean comparison DataFrame and detailed metrics dictionary.
    """
    results = {}
    table_rows = []
    
    for key, model in models.items():
        eval_result = evaluate_single_model(model, X_test, y_test)
        results[key] = eval_result
        
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        acc = eval_result['accuracy']
        prec = eval_result['precision']
        rec = eval_result['recall']
        f1 = eval_result['f1']
        auc = eval_result['roc_auc']

        table_rows.append({
            'Model Key': key,
            'Model': display_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'Accuracy (%)': f"{acc * 100:.2f}%",
            'Precision (%)': f"{prec * 100:.2f}%",
            'Recall (%)': f"{rec * 100:.2f}%",
            'F1 (%)': f"{f1 * 100:.2f}%",
            'ROC-AUC (%)': f"{auc * 100:.2f}%"
        })
        
    comparison_df = pd.DataFrame(table_rows).sort_values(by='F1', ascending=False).reset_index(drop=True)
    return comparison_df, results


def get_percentage_summary_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a clean dataframe displaying only model names and percentage metrics."""
    pct_cols = ['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'ROC-AUC (%)']
    return comparison_df[pct_cols].copy()


def save_evaluation_outputs(
    results: Dict[str, Any],
    comparison_df: pd.DataFrame,
    output_dir: str
):
    """
    Saves metrics JSON, confusion matrix plots, ROC curve plots, and percentage CSVs.
    """
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    roc_dir = os.path.join(output_dir, 'roc_curves')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    
    # Save standard & percentage comparison dataframes
    numeric_cols = ['Model Key', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    comparison_df[numeric_cols].to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    pct_df = get_percentage_summary_table(comparison_df)
    pct_df.to_csv(os.path.join(output_dir, 'model_comparison_percentages.csv'), index=False)
    
    # Clean results for JSON serialization with explicit percentages
    serializable_metrics = {}
    for key, data in results.items():
        serializable_metrics[key] = {
            'model_name': MODEL_DISPLAY_NAMES.get(key, key),
            'accuracy': data['accuracy'],
            'precision': data['precision'],
            'recall': data['recall'],
            'f1': data['f1'],
            'roc_auc': data['roc_auc'],
            'percentages': {
                'accuracy': f"{data['accuracy'] * 100:.2f}%",
                'precision': f"{data['precision'] * 100:.2f}%",
                'recall': f"{data['recall'] * 100:.2f}%",
                'f1': f"{data['f1'] * 100:.2f}%",
                'roc_auc': f"{data['roc_auc'] * 100:.2f}%"
            },
            'confusion_matrix': data['confusion_matrix'],
            'roc_curve': data['roc_curve']
        }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
        
    # Plot & save individual Confusion Matrices
    for key, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        cm = np.array(data['confusion_matrix'])
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk']
        )
        plt.title(f'Confusion Matrix: {display_name}', fontsize=11, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f'{key}_confusion_matrix.png'), dpi=200)
        plt.close()
        
    # Plot & save Combined ROC Curves
    plt.figure(figsize=(8, 6))
    for key, data in results.items():
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        fpr = data['roc_curve']['fpr']
        tpr = data['roc_curve']['tpr']
        auc = data['roc_auc']
        plt.plot(fpr, tpr, label=f"{display_name} (AUC = {auc:.3f})", lw=2)
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(roc_dir, 'combined_roc_curve.png'), dpi=200)
    plt.close()
