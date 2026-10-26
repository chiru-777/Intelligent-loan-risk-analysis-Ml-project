"""
SQLite Database Integration Module for Intelligent Loan Risk Assessment.
Handles:
- SQLite connection management
- Schema initialization (loan_applicants, prediction_audit_logs)
- CSV dataset ingestion to SQLite
- Dataset querying and retrieval directly from SQLite
- Prediction audit logging (saves user inputs and model classifications to SQLite)
- Safe SQL query execution for the Streamlit database explorer
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(CURRENT_DIR, 'data', 'loan_risk.db')
CSV_PATH = os.path.join(CURRENT_DIR, 'data', 'loan_applicants.csv')


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite database connection with row access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DB_PATH):
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Main applicants table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS loan_applicants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_code TEXT UNIQUE,
            age INTEGER,
            income REAL,
            credit_score INTEGER,
            loan_amount REAL,
            employment_status TEXT,
            loan_term INTEGER,
            existing_debts REAL,
            debt_to_income_ratio REAL,
            credit_history TEXT,
            payment_history TEXT,
            assets REAL,
            previous_loans INTEGER,
            risk TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. Prediction audit log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            income REAL,
            credit_score INTEGER,
            loan_amount REAL,
            employment_status TEXT,
            loan_term INTEGER,
            existing_debts REAL,
            debt_to_income_ratio REAL,
            credit_history TEXT,
            payment_history TEXT,
            assets REAL,
            previous_loans INTEGER,
            predicted_risk TEXT,
            risk_probability REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def import_csv_to_sqlite(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> int:
    """
    Ingests records from CSV into SQLite table `loan_applicants`.
    Skips if data is already loaded.
    Returns total rows in the database table.
    """
    init_database(db_path)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV not found at {csv_path}")
        
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM loan_applicants")
    existing_count = cursor.fetchone()[0]
    
    if existing_count > 0:
        conn.close()
        return existing_count
        
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Generate unique applicant codes if not present
    if 'applicant_code' not in df.columns:
        df['applicant_code'] = [f"APP-{1000 + i}" for i in range(len(df))]
        
    # Reorder columns to match schema
    cols = [
        'applicant_code', 'age', 'income', 'credit_score', 'loan_amount',
        'employment_status', 'loan_term', 'existing_debts', 'debt_to_income_ratio',
        'credit_history', 'payment_history', 'assets', 'previous_loans', 'risk'
    ]
    
    insert_data = df[cols].copy()
    insert_data.to_sql('loan_applicants', conn, if_exists='append', index=False)
    
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM loan_applicants")
    total_count = cursor.fetchone()[0]
    conn.close()
    
    return total_count


def load_dataset_from_sqlite(db_path: str = DB_PATH) -> pd.DataFrame:
    """Loads all loan applicant records directly from SQLite as a DataFrame."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    
    # Check if empty, populate from CSV if necessary
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM loan_applicants")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(CSV_PATH):
        conn.close()
        import_csv_to_sqlite(CSV_PATH, db_path)
        conn = get_db_connection(db_path)
        
    query = """
        SELECT age, income, credit_score, loan_amount, employment_status, 
               loan_term, existing_debts, debt_to_income_ratio, credit_history, 
               payment_history, assets, previous_loans, risk
        FROM loan_applicants
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def log_prediction_to_sqlite(
    applicant_data: Dict[str, Any],
    predicted_risk: str,
    risk_probability: float,
    model_used: str = "Stacking Ensemble",
    db_path: str = DB_PATH
) -> int:
    """Logs an individual inference event into prediction_audit_logs."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO prediction_audit_logs (
            age, income, credit_score, loan_amount, employment_status,
            loan_term, existing_debts, debt_to_income_ratio, credit_history,
            payment_history, assets, previous_loans, predicted_risk,
            risk_probability, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(applicant_data.get('age', 0)),
        float(applicant_data.get('income', 0.0)),
        int(applicant_data.get('credit_score', 0)),
        float(applicant_data.get('loan_amount', 0.0)),
        str(applicant_data.get('employment_status', '')),
        int(applicant_data.get('loan_term', 0)),
        float(applicant_data.get('existing_debts', 0.0)),
        float(applicant_data.get('debt_to_income_ratio', 0.0)),
        str(applicant_data.get('credit_history', '')),
        str(applicant_data.get('payment_history', '')),
        float(applicant_data.get('assets', 0.0)),
        int(applicant_data.get('previous_loans', 0)),
        str(predicted_risk),
        float(risk_probability),
        str(model_used)
    ))
    
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def get_prediction_audit_logs(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves the most recent inference audit logs from SQLite."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    query = f"SELECT * FROM prediction_audit_logs ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def execute_safe_query(sql_query: str, db_path: str = DB_PATH) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Executes a SELECT query safely against the SQLite database.
    Prevents destructive operations (DROP, DELETE, UPDATE, INSERT, ALTER).
    """
    cleaned_query = sql_query.strip()
    first_word = cleaned_query.split()[0].upper() if cleaned_query else ""
    
    if first_word != "SELECT" and first_word != "PRAGMA":
        return None, "Only SELECT or PRAGMA queries are allowed in the read-only explorer."
        
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(cleaned_query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def get_database_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Returns metadata and statistics about the SQLite database."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM loan_applicants")
    applicants_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prediction_audit_logs")
    audit_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
    
    conn.close()
    
    file_size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    file_size_kb = round(file_size_bytes / 1024, 2)
    
    return {
        'db_path': db_path,
        'file_size_kb': file_size_kb,
        'tables': tables,
        'applicants_count': applicants_count,
        'audit_logs_count': audit_count
    }
