"""
Preprocessing Module for Credit Scoring System.

Builds and exports Scikit-Learn data transformation pipelines using StandardScaler
for numerical features and OneHotEncoder for categorical features.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    "Age", "Income", "Loan_Amount", "Existing_Debt", "Number_of_Loans", "Credit_Score_Value"
]

CATEGORICAL_FEATURES = [
    "Employment_Status", "Credit_History", "Payment_History"
]

TARGET_COLUMN = "Creditworthy"


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for numerical & categorical preprocessing.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, save_path: Path) -> None:
    """Saves fitted preprocessor to disk using joblib."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)


def load_preprocessor(load_path: Path) -> ColumnTransformer:
    """Loads preprocessor artifact from disk using joblib."""
    return joblib.load(load_path)
