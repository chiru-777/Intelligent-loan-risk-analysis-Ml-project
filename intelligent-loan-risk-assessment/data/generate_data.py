"""
Script to generate a realistic loan applicant dataset.
Includes realistic credit risk factors, slight noise, duplicate records,
and missing values to validate the preprocessing pipeline.
"""
import numpy as np
import pandas as pd

def generate_loan_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Age: 21 to 68
    age = np.random.randint(21, 69, size=n_samples)
    
    # 2. Employment status
    emp_choices = ['Employed', 'Self-Employed', 'Unemployed']
    emp_probs = [0.72, 0.20, 0.08]
    employment_status = np.random.choice(emp_choices, size=n_samples, p=emp_probs)
    
    # 3. Income based on employment and age
    income = np.zeros(n_samples)
    for i in range(n_samples):
        if employment_status[i] == 'Unemployed':
            inc = np.random.normal(18000, 5000)
        elif employment_status[i] == 'Self-Employed':
            inc = np.random.lognormal(mean=11.1, sigma=0.55)
        else: # Employed
            exp_factor = (age[i] - 21) * 600
            inc = np.random.normal(55000 + exp_factor, 18000)
        income[i] = max(12000, min(inc, 250000))
        
    income = np.round(income, -2) # Round to nearest 100
    
    # 4. Credit score: 300 to 850 (Normal-ish distribution around 680)
    credit_score = np.random.normal(675, 85, size=n_samples)
    # Adjust for employment
    credit_score = np.where(employment_status == 'Unemployed', credit_score - 70, credit_score)
    credit_score = np.clip(np.round(credit_score), 320, 850).astype(int)
    
    # 5. Loan Amount: $3,000 to $95,000 correlated with income
    loan_amount = np.zeros(n_samples)
    for i in range(n_samples):
        max_borrow = income[i] * np.random.uniform(0.15, 0.75)
        loan_amt = np.random.normal(max_borrow * 0.8, 10000)
        loan_amount[i] = max(3000, min(loan_amt, 95000))
    loan_amount = np.round(loan_amount, -2)
    
    # 6. Loan Term in months: 12, 24, 36, 48, 60
    loan_term = np.random.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.12, 0.22, 0.36, 0.18, 0.12])
    
    # 7. Existing debts: correlated with income & age
    existing_debts = np.zeros(n_samples)
    for i in range(n_samples):
        debts = np.random.exponential(scale=income[i] * 0.25)
        existing_debts[i] = max(0, min(debts, 80000))
    existing_debts = np.round(existing_debts, -2)
    
    # 8. Debt-to-Income (DTI) ratio
    # Realistic formulation: (existing_debts + (loan_amount / (loan_term / 12))) / (income + 1e-5)
    annual_loan_repay = (loan_amount / loan_term) * 12
    dti = (existing_debts * 0.12 + annual_loan_repay) / (income + 1e-5)
    dti = np.round(np.clip(dti, 0.05, 0.95), 4)
    
    # 9. Credit History: Poor, Fair, Good, Excellent
    credit_history = []
    for cs in credit_score:
        if cs < 580:
            credit_history.append('Poor')
        elif cs < 670:
            credit_history.append('Fair')
        elif cs < 750:
            credit_history.append('Good')
        else:
            credit_history.append('Excellent')
    credit_history = np.array(credit_history)
    
    # Add minor noise in credit history rating
    flip_idx = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
    for idx in flip_idx:
        credit_history[idx] = np.random.choice(['Fair', 'Good'])
        
    # 10. Payment History: On-Time, Delayed, Missed
    payment_history = []
    for i in range(n_samples):
        cs = credit_score[i]
        if cs < 580:
            p = [0.15, 0.40, 0.45]
        elif cs < 670:
            p = [0.45, 0.40, 0.15]
        elif cs < 750:
            p = [0.80, 0.16, 0.04]
        else:
            p = [0.94, 0.05, 0.01]
        payment_history.append(np.random.choice(['On-Time', 'Delayed', 'Missed'], p=p))
    payment_history = np.array(payment_history)
    
    # 11. Assets: savings, property, investments
    assets = np.zeros(n_samples)
    for i in range(n_samples):
        asset_val = income[i] * np.random.uniform(0.2, 2.5) + (age[i] - 21) * 2000
        if employment_status[i] == 'Unemployed':
            asset_val *= 0.3
        assets[i] = max(0, min(asset_val, 350000))
    assets = np.round(assets, -2)
    
    # 12. Previous loans count: 0 to 6
    previous_loans = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.32, 0.22, 0.12, 0.06, 0.03])
    
    # 13. Determine Risk Label based on realistic financial scoring
    # Log-odds of High Risk
    log_odds = np.zeros(n_samples)
    
    # Credit score impact (Strongest standard predictor)
    log_odds += (660 - credit_score) / 60.0  # <660 increases risk, >660 decreases risk
    
    # DTI impact (High DTI increases risk)
    log_odds += (dti - 0.35) * 4.5
    
    # Payment history impact
    for i in range(n_samples):
        if payment_history[i] == 'Missed':
            log_odds[i] += 1.8
        elif payment_history[i] == 'Delayed':
            log_odds[i] += 0.7
        else: # On-Time
            log_odds[i] -= 0.9
            
    # Employment impact
    for i in range(n_samples):
        if employment_status[i] == 'Unemployed':
            log_odds[i] += 2.2
        elif employment_status[i] == 'Self-Employed':
            log_odds[i] += 0.25
        else:
            log_odds[i] -= 0.4
            
    # Loan amount relative to assets & income
    loan_to_income = loan_amount / (income + 1e-5)
    log_odds += (loan_to_income - 0.4) * 1.8
    
    # Asset buffer
    asset_to_loan = assets / (loan_amount + 1e-5)
    log_odds -= np.clip((asset_to_loan - 1.0) * 0.4, -1.2, 1.2)
    
    # Previous loans: experienced repayers who have 1-3 previous loans and on-time history have lower risk
    for i in range(n_samples):
        if previous_loans[i] >= 1 and payment_history[i] == 'On-Time':
            log_odds[i] -= 0.35
        elif previous_loans[i] >= 3 and payment_history[i] != 'On-Time':
            log_odds[i] += 0.5
            
    # Age factor: very young with low assets slightly riskier
    log_odds += np.where((age < 25) & (assets < 10000), 0.4, 0.0)
    
    # Add realistic stochastic noise
    log_odds += np.random.normal(0, 0.45, size=n_samples)
    
    # Sigmoid to get probability of High Risk
    prob_high_risk = 1.0 / (1.0 + np.exp(-log_odds))
    
    risk = np.where(prob_high_risk >= 0.5, 'High Risk', 'Low Risk')
    
    df = pd.DataFrame({
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
        'previous_loans': previous_loans,
        'risk': risk
    })
    
    # Introduce ~15 duplicate rows to test duplicate removal
    dup_indices = np.random.choice(n_samples, size=15, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Introduce ~35 missing values across various columns to test missing value imputation
    nan_cols = ['income', 'credit_score', 'existing_debts', 'payment_history', 'assets']
    for col in nan_cols:
        nan_idx = np.random.choice(len(df), size=7, replace=False)
        df.loc[nan_idx, col] = np.nan
        
    return df

if __name__ == '__main__':
    df = generate_loan_data(3000, random_state=42)
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'loan_applicants.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print("Class distribution:")
    print(df['risk'].value_counts(dropna=False))
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicates count:")
    print(df.duplicated().sum())
