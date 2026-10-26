"""
Machine Learning Models and Ensemble Architecture Module.
Defines, instantiates, and trains:
1. Support Vector Machine (SVM)
2. K-Nearest Neighbors (KNN)
3. Random Forest (RF)
4. Gradient Boosting (GB)
5. Soft Voting Classifier
6. Stacking Classifier
"""

import os
import joblib
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression

MODEL_FILENAMES = {
    'svm': 'svm_model.pkl',
    'knn': 'knn_model.pkl',
    'random_forest': 'random_forest_model.pkl',
    'gradient_boosting': 'gradient_boosting_model.pkl',
    'soft_voting': 'voting_model.pkl',
    'stacking': 'stacking_model.pkl'
}

MODEL_DISPLAY_NAMES = {
    'svm': 'Support Vector Machine (SVM)',
    'knn': 'K-Nearest Neighbors (KNN)',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'soft_voting': 'Soft Voting Ensemble',
    'stacking': 'Stacking Ensemble'
}


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate the 4 base machine learning models."""
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=random_state
    )
    
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    )
    
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        random_state=random_state
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        random_state=random_state
    )
    
    return {
        'svm': svm,
        'knn': knn,
        'random_forest': rf,
        'gradient_boosting': gb
    }


def get_all_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate all 6 models including base classifiers and both ensembles.
    """
    base_models = get_base_models(random_state=random_state)
    
    # 5. Soft Voting Ensemble
    voting = VotingClassifier(
        estimators=[
            ('svm', get_base_models(random_state)['svm']),
            ('knn', get_base_models(random_state)['knn']),
            ('rf', get_base_models(random_state)['random_forest']),
            ('gb', get_base_models(random_state)['gradient_boosting'])
        ],
        voting='soft'
    )
    
    # 6. Stacking Classifier with Logistic Regression Meta-Learner
    stacking = StackingClassifier(
        estimators=[
            ('svm', get_base_models(random_state)['svm']),
            ('knn', get_base_models(random_state)['knn']),
            ('rf', get_base_models(random_state)['random_forest']),
            ('gb', get_base_models(random_state)['gradient_boosting'])
        ],
        final_estimator=LogisticRegression(random_state=random_state),
        cv=5
    )
    
    all_models = base_models.copy()
    all_models['soft_voting'] = voting
    all_models['stacking'] = stacking
    
    return all_models


def save_models(models: Dict[str, Any], output_dir: str):
    """Save all trained model artifacts into the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    for key, model in models.items():
        filename = MODEL_FILENAMES.get(key, f"{key}_model.pkl")
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
        

def load_models(models_dir: str) -> Dict[str, Any]:
    """Load all saved models from disk."""
    loaded_models = {}
    for key, filename in MODEL_FILENAMES.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            loaded_models[key] = joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}. Run train.py first.")
    return loaded_models
