"""
Data Loader Module for Credit Scoring System.

Generates a realistic synthetic financial dataset (if missing) and loads it for
model training, evaluation, and data exploration.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from backend.config.settings import DATASET_PATH, DATA_DIR


def generate_synthetic_credit_data(num_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic credit scoring dataset with financial indicators.
    """
    np.random.seed(random_seed)

    ages = np.random.randint(21, 68, size=num_samples)
    incomes = np.random.randint(25000, 140000, size=num_samples)

    employment_options = ["Employed", "Self-Employed", "Unemployed"]
    employment_probs = [0.70, 0.22, 0.08]
    employment_status = np.random.choice(employment_options, size=num_samples, p=employment_probs)

    loan_amounts = np.random.randint(3000, 45000, size=num_samples)
    existing_debts = np.random.randint(500, 35000, size=num_samples)

    credit_history_options = ["Good", "Fair", "Poor"]
    credit_history_probs = [0.55, 0.30, 0.15]
    credit_history = np.random.choice(credit_history_options, size=num_samples, p=credit_history_probs)

    number_of_loans = np.random.randint(1, 7, size=num_samples)

    payment_history_options = ["On-Time", "Delayed", "Defaulted"]
    payment_history_probs = [0.65, 0.25, 0.10]
    payment_history = np.random.choice(payment_history_options, size=num_samples, p=payment_history_probs)

    credit_scores = np.random.randint(320, 840, size=num_samples)

    # Risk Scoring Formula to derive Target (Creditworthy)
    score = np.zeros(num_samples)

    dti_ratio = (existing_debts + loan_amounts) / incomes
    score += np.where(dti_ratio < 0.4, 25, np.where(dti_ratio < 0.7, 10, -20))

    score += (credit_scores - 300) / 10

    score += np.where(payment_history == "On-Time", 20, np.where(payment_history == "Delayed", -10, -35))
    score += np.where(credit_history == "Good", 15, np.where(credit_history == "Fair", 5, -15))
    score += np.where(employment_status == "Employed", 10, np.where(employment_status == "Self-Employed", 5, -20))

    score += np.random.normal(0, 5, size=num_samples)

    target = (score >= 45).astype(int)

    df = pd.DataFrame({
        "Age": ages,
        "Income": incomes,
        "Employment_Status": employment_status,
        "Loan_Amount": loan_amounts,
        "Existing_Debt": existing_debts,
        "Credit_History": credit_history,
        "Number_of_Loans": number_of_loans,
        "Payment_History": payment_history,
        "Credit_Score_Value": credit_scores,
        "Creditworthy": target
    })

    return df


def load_credit_data() -> pd.DataFrame:
    """
    Loads the credit scoring dataset. Generates dataset if file does not exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        df = generate_synthetic_credit_data()
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)

    return df
