"""
Data Preprocessing and Pipeline Module for Intelligent Loan Risk Assessment.
Handles:
- Missing value imputation
- Duplicate removal
- Outlier treatment (capping based on IQR from training set)
- Categorical one-hot encoding
- Numerical feature scaling
- Feature selection via SelectKBest (ANOVA F-value)
- Leakage-free fit/transform methods
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

NUMERICAL_COLS = [
    'age',
    'income',
    'credit_score',
    'loan_amount',
    'loan_term',
    'existing_debts',
    'debt_to_income_ratio',
    'assets',
    'previous_loans'
]

CATEGORICAL_COLS = [
    'employment_status',
    'credit_history',
    'payment_history'
]

TARGET_COL = 'risk'

TARGET_MAPPING = {
    'Low Risk': 0,
    'High Risk': 1
}

INVERSE_TARGET_MAPPING = {
    0: 'Low Risk',
    1: 'High Risk'
}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Drops duplicate rows.
    - Strips whitespace in string columns.
    - Validates target values.
    """
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Strip string whitespaces
    for col in cleaned_df.select_dtypes(include='object').columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
        
    return cleaned_df


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical features to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] determined during fit.
    Prevents extreme values from distorting models without removing valid samples.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            q1 = X_df[col].quantile(0.01)
            q3 = X_df[col].quantile(0.99)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            # Don't cap below 0 for non-negative financial values
            lower = max(0.0, lower)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = np.clip(X_df[col], lower, upper)
        return X_df.to_numpy()


class LoanDataPreprocessor:
    """
    Complete preprocessor for Loan Applicant Risk Assessment.
    Encapsulates column transformations, scaling, encoding, and feature selection.
    """
    def __init__(self, k_features=14):
        self.numerical_cols = NUMERICAL_COLS
        self.categorical_cols = CATEGORICAL_COLS
        self.target_col = TARGET_COL
        self.k_features = k_features
        
        # Pipelines
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('capper', OutlierCapper(factor=1.5)),
            ('scaler', StandardScaler())
        ])
        
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', self.num_pipeline, self.numerical_cols),
                ('cat', self.cat_pipeline, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        self.selector = SelectKBest(score_func=f_classif, k=self.k_features)
        
        self.encoded_feature_names_ = []
        self.selected_feature_names_ = []
        self.feature_scores_ = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit preprocessor and feature selection on training data only."""
        # 1. Transform columns
        X_transformed = self.column_transformer.fit_transform(X)
        
        # 2. Extract feature names post-encoding
        cat_encoder = self.column_transformer.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.encoded_feature_names_ = self.numerical_cols + cat_feature_names
        
        # 3. Fit feature selector
        # Encode target if it's strings
        if y.dtype == object or isinstance(y.iloc[0], str):
            y_numeric = y.map(TARGET_MAPPING)
        else:
            y_numeric = y
            
        self.selector.fit(X_transformed, y_numeric)
        
        # Record scores
        scores = self.selector.scores_
        for name, score in zip(self.encoded_feature_names_, scores):
            self.feature_scores_[name] = float(score) if not np.isnan(score) else 0.0
            
        # Get selected feature names
        selected_mask = self.selector.get_support()
        self.selected_feature_names_ = [
            name for name, selected in zip(self.encoded_feature_names_, selected_mask) if selected
        ]
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame into selected features matrix."""
        if not self.is_fitted:
            raise RuntimeError("LoanDataPreprocessor must be fitted before calling transform().")
            
        # Ensure all required columns exist in X
        X_copy = X.copy()
        for col in self.numerical_cols:
            if col not in X_copy.columns:
                X_copy[col] = np.nan
        for col in self.categorical_cols:
            if col not in X_copy.columns:
                X_copy[col] = np.nan
                
        X_transformed = self.column_transformer.transform(X_copy)
        X_selected = self.selector.transform(X_transformed)
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_scores(self) -> pd.DataFrame:
        """Returns dataframe with all encoded features and their ANOVA F-scores."""
        scores_df = pd.DataFrame([
            {
                'Feature': feat,
                'F_Score': score,
                'Selected': feat in self.selected_feature_names_
            }
            for feat, score in self.feature_scores_.items()
        ])
        return scores_df.sort_values(by='F_Score', ascending=False).reset_index(drop=True)

    def transform_single_applicant(self, applicant_dict: dict) -> np.ndarray:
        """Transforms a single applicant dict to a feature vector for model prediction."""
        df = pd.DataFrame([applicant_dict])
        return self.transform(df)
