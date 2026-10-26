# Intelligent Loan Risk Assessment System

**Aditya University**  
*Department of Artificial Intelligence and Machine Learning*  

**Project Team:**
* **E. Chiranjeevi** — `24B11AI105`
* **E. Madhu Sudhan Reddy** — `24B11AI022`
* **B. D V Sai Ajay** — `24B11AI022`
* **M. Surya Maha Teja** — `24B11AI251`

---

## Project Overview

The **Intelligent Loan Risk Assessment System** is an end-to-end, production-style Machine Learning application designed to predict and explain credit default risk for loan applicants. Rather than relying on traditional rigid heuristic rules or single-model classification, this system implements a multi-model comparative architecture paired with advanced ensemble learning and Explainable AI (XAI).

The application classifies loan applicants into:
* **Low Risk** (Eligible for loan approval / standard underwriting)
* **High Risk** (Requires mitigation, higher collateral, or rejection)

The system features an **Interactive What-If Risk Simulator**, empowering financial officers and borrowers to dynamically test credit restructuring scenarios in real time.

---

## Features

1. **Multi-Model Predictive Engine:**
   * Trains and compares 4 base algorithms: **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, **Random Forest (RF)**, and **Gradient Boosting (GB)**.
2. **Advanced Ensemble Learning:**
   * **Soft Voting Classifier:** Aggregates calibrated risk probabilities across all 4 base models.
   * **Stacking Classifier:** Uses a 5-fold cross-validated meta-learner (Logistic Regression) to synthesize base model outputs into an optimal final prediction.
3. **Interactive What-If Risk Simulator:**
   * Major innovative feature allowing users to modify income, credit score, loan amount, debt-to-income (DTI) ratio, employment status, and loan terms.
   * Dynamically contrasts original risk vs. simulated risk with probability shift indicators ($\Delta\%$) and gauge comparisons.
4. **Explainable AI using SHAP:**
   * Demystifies black-box model decisions using Shapley Additive Explanations.
   * Provides both global feature importance and local applicant waterfall plots with plain-English contextual explanations.
5. **Comprehensive Preprocessing & Feature Selection:**
   * Automated missing-value imputation, duplicate removal, outlier capping (IQR-based), categorical one-hot encoding, feature scaling, and statistical feature selection (SelectKBest ANOVA F-scores) with strict leakage prevention.
6. **Executive Web Dashboard:**
   * High-level portfolio analytics, risk distributions, income vs. loan scatter plots, and credit score distributions built with Plotly and Streamlit.
7. **Production-Ready & Fully Reproducible:**
   * Modular architecture with separate modules for preprocessing, training, inference, evaluation, visualization, and explainability.

---

## Technologies Used

* **Database:** SQLite 3 (`data/loan_risk.db`)
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Web Application:** Streamlit
* **Visualizations:** Plotly, Matplotlib, Seaborn
* **Model Serialization:** Joblib
* **Experimentation:** Jupyter Notebook (`notebooks/loan_risk_analysis.ipynb`)

---

## SQLite Database Integration

The system integrates a relational **SQLite 3** database (`data/loan_risk.db`) that serves as the single source of truth for credit records and model auditing:
* **`loan_applicants` Table:** Stores the cleaned applicant records with full demographic, financial, and credit history attributes.
* **`prediction_audit_logs` Table:** Automatically records every prediction executed in the Streamlit web app (applicant profile, timestamp, predicted risk, and calibrated probability).
* **Live SQL Explorer:** Built directly into the Streamlit application under *Dataset Insights → SQLite Database Explorer*, allowing users to inspect tables and execute read-only SQL queries.

Initialize or re-sync the database anytime with:
```bash
python data/init_database.py
```

---

## Dataset

* **Source:** Synthetic banking dataset calibrated with realistic demographic, financial, and credit risk distributions (`data/loan_applicants.csv`).
* **Sample Size:** 3,015 raw records (3,000 unique records post-cleaning).
* **Attributes:**
  * `age`: Applicant age (21–68)
  * `income`: Annual income in USD ($12,000–$250,000)
  * `credit_score`: FICO credit score (320–850)
  * `loan_amount`: Requested principal ($3,000–$95,000)
  * `loan_term`: Repayment duration (12, 24, 36, 48, 60 months)
  * `employment_status`: `Employed`, `Self-Employed`, `Unemployed`
  * `existing_debts`: Total current financial obligations ($0–$80,000)
  * `debt_to_income_ratio`: DTI ratio (0.05–0.95)
  * `credit_history`: Rating (`Poor`, `Fair`, `Good`, `Excellent`)
  * `payment_history`: Repayment track record (`On-Time`, `Delayed`, `Missed`)
  * `assets`: Collateral and liquid assets ($0–$350,000)
  * `previous_loans`: Number of prior credit lines (0–5)
  * `risk`: Target label (`Low Risk`, `High Risk`)

---

## Machine Learning Models & Ensemble Learning

The architecture consists of four distinct algorithmic families and two ensemble layers:

```
                     Loan Applicant Dataset
                               │
                               ▼
                       Data Preprocessing
        (Cleaning, Imputation, Capping, Encoding, Scaling)
                               │
                               ▼
                       Feature Selection
                         (SelectKBest)
                               │
                               ▼
                         Training Data
                               │
   ┌───────────────┬───────────┴───┬────────────────┬─────────────────┐
   ▼               ▼               ▼                ▼                 ▼
  SVM             KNN        Random Forest    Gradient Boosting     ...
   │               │               │                │
   └───────────────┴───────────┬───┴────────────────┘
                               │
                               ▼
                         Ensemble Layer
                       ┌───────┴───────┐
                       ▼               ▼
                  Soft Voting       Stacking
                  Classifier       Classifier
                       └───────┬───────┘
                               │
                               ▼
                      Final Risk Prediction
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Explainable AI (SHAP)        What-If Risk Simulator
    (Marginal Feature Drivers)    (Dynamic Scenario Testing)
```

### Actual Evaluation Performance (Holdout Test Cohort - 600 Applicants)

| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | **96.67%** | **95.93%** | **88.72%** | **0.9219** | **0.9947** |
| **Soft Voting Ensemble** | **96.50%** | **95.90%** | **87.97%** | **0.9176** | **0.9947** |
| **Gradient Boosting** | **96.17%** | **95.83%** | **86.47%** | **0.9091** | **0.9939** |
| **Support Vector Machine (SVM)** | **96.00%** | **93.60%** | **87.97%** | **0.9070** | **0.9920** |
| **Random Forest** | **95.83%** | **96.55%** | **84.21%** | **0.8996** | **0.9928** |
| **K-Nearest Neighbors (KNN)** | **94.00%** | **95.33%** | **76.69%** | **0.8500** | **0.9797** |

*Note: All performance scores are calculated dynamically from actual model inference on holdout test data.*

---

## Explainable AI (SHAP)

The Explainable AI module computes exact Shapley marginal contributions:
* **Global Importance:** Unveils macro factors driving risk across the portfolio (Credit Score, Debt-to-Income, Repayment History, Income).
* **Local Waterfall Analysis:** Explains individual credit decisions with directional drivers (e.g., `Credit Score (-0.42 log-odds)` lowered risk, while `DTI Ratio (+0.31 log-odds)` elevated risk).
* **Plain-English Explanations:** Translates statistical values into actionable, regulator-friendly insights.

---

## What-If Risk Simulator

The **What-If Risk Simulator** addresses a key limitation of static prediction tools:
* Side-by-side interactive controls let users adjust one or multiple parameters.
* Compares baseline risk probability to the new simulated scenario.
* Provides visual gauge indicators and probability delta calculations ($\pm \Delta\%$).
* *Disclaimer: For educational and decision-support demonstration purposes only; does not guarantee loan approval.*

---

## Project Structure

```
intelligent-loan-risk-assessment/
│
├── app.py                      # Main interactive Streamlit application
├── train.py                    # Complete model training and evaluation script
├── predict.py                  # Standalone inference and batch prediction engine
├── requirements.txt            # Dependency configuration
├── README.md                   # Complete documentation
├── .gitignore                  # Git ignore definitions
│
├── data/
│   ├── loan_applicants.csv     # Cleaned loan applicant dataset
│   └── generate_data.py        # Dataset generation & validation script
│
├── models/
│   ├── svm_model.pkl           # Trained Support Vector Machine
│   ├── knn_model.pkl           # Trained K-Nearest Neighbors
│   ├── random_forest_model.pkl # Trained Random Forest
│   ├── gradient_boosting_model.pkl # Trained Gradient Boosting
│   ├── voting_model.pkl        # Trained Soft Voting Ensemble
│   ├── stacking_model.pkl      # Trained Stacking Ensemble
│   └── preprocessing.pkl       # Serialized preprocessing and feature selection pipeline
│
├── notebooks/
│   ├── loan_risk_analysis.ipynb # Interactive Jupyter exploration & analysis
│   └── generate_notebook.py    # Script generating the notebook
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # Outlier capper, ColumnTransformer, SelectKBest pipeline
│   ├── models.py               # Model definitions, ensembles, and factory methods
│   ├── evaluation.py           # Metric calculation, ROC curves, confusion matrices
│   ├── explainability.py       # SHAP explainers and natural language generator
│   └── visualization.py        # Plotly chart creators for dashboard and simulator
│
└── outputs/
    ├── confusion_matrices/     # Exported confusion matrix heatmaps
    ├── roc_curves/             # Exported multi-model ROC curves
    ├── feature_importance/     # Exported SHAP global summary plot
    ├── metrics.json            # Machine-readable evaluation metrics
    └── model_comparison.csv    # Tabular model benchmark results
```

---

## Installation & Setup

Clone or navigate to the project directory:

```bash
cd intelligent-loan-risk-assessment
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Models

To execute data cleaning, train all 6 machine learning models, compute performance metrics, and save all artifacts:

```bash
python train.py
```

---

## Running the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The web application will open in your default browser at `http://localhost:8501`.

---

## Example Workflow

1. **Dashboard Inspection:** View overall portfolio default rates, credit score distribution, and income vs. loan amount patterns.
2. **Predict Risk:** Enter applicant financial parameters or click *"Load Low-Risk Sample Profile"* / *"Load High-Risk Sample Profile"*. Click **Predict Risk** to view individual and ensemble model outputs.
3. **What-If Simulation:** Navigate to the **What-If Simulator**, modify parameters (e.g., increase credit score from 540 to 680 or lower loan amount), and watch the risk probability gauge dynamically update.
4. **Explainable AI:** View the SHAP waterfall chart and read plain-English explanations detailing why the applicant received their risk classification.
5. **Model Benchmarking:** Inspect the **Model Comparison** page to analyze the ROC curve, confusion matrix heatmaps, and metric tables across all 6 architectures.

---

## Future Enhancements

* Support for multi-product loan types (auto loans, mortgages, revolving credit lines).
* Real-time bank statement API integration.
* Time-series payment delinquency forecasting.
* Mobile-optimized frontend interface.
