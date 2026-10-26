"""
Database Setup & Ingestion Script.
Initializes SQLite database and populates it with records from data/loan_applicants.csv.
"""

import os
import sys

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.database import import_csv_to_sqlite, get_database_stats, DB_PATH, CSV_PATH

def main():
    print("=" * 70)
    print("  INITIALIZING SQLITE DATABASE FOR INTELLIGENT LOAN RISK SYSTEM  ")
    print("=" * 70)
    print(f"[*] Target SQLite Database: {DB_PATH}")
    print(f"[*] Source CSV File       : {CSV_PATH}")
    
    total_records = import_csv_to_sqlite()
    stats = get_database_stats()
    
    print("\n[+] Database successfully connected and initialized!")
    print(f"  • Total Applicant Records Ingested : {stats['applicants_count']:,}")
    print(f"  • Database File Size               : {stats['file_size_kb']} KB")
    print(f"  • Tables Present                   : {', '.join(stats['tables'])}")
    print(f"  • Audit Logs Table Ready           : {stats['audit_logs_count']} records")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
