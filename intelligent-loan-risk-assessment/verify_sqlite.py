"""
Verification script demonstrating live SQLite connection and querying.
"""
import sqlite3
import os

db_path = 'data/loan_risk.db'

print(f"1. Database file exists: {os.path.exists(db_path)}")
print(f"2. Database path       : {os.path.abspath(db_path)}")
print(f"3. Database file size  : {os.path.getsize(db_path):,} bytes")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Check tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cur.fetchall()]
print(f"4. Tables in SQLite    : {tables}")

# Count records
cur.execute("SELECT COUNT(*) FROM loan_applicants;")
count = cur.fetchone()[0]
print(f"5. Total records       : {count:,} applicants")

# Group by risk
cur.execute("SELECT risk, COUNT(*), ROUND(AVG(income), 2), ROUND(AVG(credit_score), 1) FROM loan_applicants GROUP BY risk;")
rows = cur.fetchall()
print("\n6. SQL Aggregation Output (GROUP BY risk):")
for r in rows:
    print(f"   • Risk: {r[0]:<10} | Count: {r[1]:<5} | Avg Income: ${r[2]:,.2f} | Avg Credit Score: {r[3]}")

# Sample records
cur.execute("SELECT applicant_code, age, income, credit_score, loan_amount, employment_status, risk FROM loan_applicants LIMIT 2;")
samples = cur.fetchall()
print("\n7. Sample Records directly from SQLite:")
for s in samples:
    print(f"   • Code: {s[0]}, Age: {s[1]}, Income: ${s[2]:,.0f}, Credit: {s[3]}, Loan: ${s[4]:,.0f}, Status: {s[5]}, Risk: {s[6]}")

conn.close()
