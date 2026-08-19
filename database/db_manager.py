"""
Database Manager Module for Credit Scoring System.

Handles SQLite database initialization, record insertion, search, and historical evaluation audits.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

DATABASE_DIR = Path(__file__).resolve().parent
DB_PATH = DATABASE_DIR / "credit_records.db"


class CreditDatabaseManager:
    """
    SQLite Database Manager for persisting applicant evaluation records.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes the database schema if table does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credit_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    applicant_name TEXT NOT NULL,
                    age INTEGER,
                    employment_status TEXT,
                    income REAL,
                    existing_debt REAL,
                    loan_amount REAL,
                    number_of_loans INTEGER,
                    credit_score_value INTEGER,
                    credit_history TEXT,
                    payment_history TEXT,
                    prediction INTEGER,
                    status TEXT,
                    probability_pct REAL,
                    risk_level TEXT,
                    dti_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def insert_evaluation(self, applicant_data: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Inserts a new credit evaluation record into SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO credit_evaluations (
                    applicant_name, age, employment_status, income, existing_debt,
                    loan_amount, number_of_loans, credit_score_value, credit_history,
                    payment_history, prediction, status, probability_pct, risk_level,
                    dti_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                applicant_data.get("applicant_name", "Anonymous"),
                applicant_data.get("Age", 35),
                applicant_data.get("Employment_Status", "Employed"),
                applicant_data.get("Income", 0.0),
                applicant_data.get("Existing_Debt", 0.0),
                applicant_data.get("Loan_Amount", 0.0),
                applicant_data.get("Number_of_Loans", 1),
                applicant_data.get("Credit_Score_Value", 650),
                applicant_data.get("Credit_History", "Good"),
                applicant_data.get("Payment_History", "On-Time"),
                result.get("prediction", 0),
                result.get("status", "Unknown"),
                result.get("probability_pct", 0.0),
                result.get("risk_level", "Unknown"),
                result.get("dti_ratio", 0.0),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cursor.lastrowid

    def fetch_all_evaluations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetches recent credit evaluation records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM credit_evaluations ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_evaluations(self, query: str) -> List[Dict[str, Any]]:
        """Searches evaluation records by applicant name or status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            search_pattern = f"%{query}%"
            cursor.execute("""
                SELECT * FROM credit_evaluations
                WHERE applicant_name LIKE ? OR status LIKE ? OR risk_level LIKE ?
                ORDER BY id DESC
            """, (search_pattern, search_pattern, search_pattern))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM credit_evaluations")
            total_evals = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM credit_evaluations WHERE prediction = 1")
            approved_evals = cursor.fetchone()[0]

            denied_evals = total_evals - approved_evals

            return {
                "total": total_evals,
                "approved": approved_evals,
                "denied": denied_evals
            }
