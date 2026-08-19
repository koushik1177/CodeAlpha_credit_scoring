"""
Prediction & Inference Module for Credit Scoring System.

Loads pre-trained model and preprocessor artifacts to evaluate creditworthiness
for new loan applicants in real time.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

from backend.config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH


class CreditScoringPredictor:
    """
    Inference Engine for evaluating creditworthiness of loan applicants.
    """

    def __init__(self, model_path: Path = BEST_MODEL_PATH, preprocessor_path: Path = PREPROCESSOR_PATH):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads fitted model and preprocessor pipeline from disk."""
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            raise FileNotFoundError(
                "Model or Preprocessor artifacts not found. Please run model training first."
            )
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict_applicant(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single applicant's data and returns prediction details.
        """
        df_input = pd.DataFrame([applicant_data])
        X_trans = self.preprocessor.transform(df_input)

        prediction = int(self.model.predict(X_trans)[0])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_trans)[0][1])
        else:
            proba = float(prediction)

        probability_pct = round(proba * 100, 2)

        if prediction == 1:
            status = "Creditworthy (Approved)"
            if probability_pct >= 75.0:
                risk_level = "Low Risk"
            else:
                risk_level = "Moderate Risk"
        else:
            status = "Not Creditworthy (Denied)"
            if probability_pct <= 25.0:
                risk_level = "High Risk"
            else:
                risk_level = "Elevated Risk"

        income = float(applicant_data.get("Income", 1))
        existing_debt = float(applicant_data.get("Existing_Debt", 0))
        loan_amount = float(applicant_data.get("Loan_Amount", 0))
        credit_score = int(applicant_data.get("Credit_Score_Value", 650))
        payment_hist = applicant_data.get("Payment_History", "On-Time")

        dti_ratio = round(((existing_debt + loan_amount) / max(income, 1.0)) * 100, 1)

        explanations = []
        if dti_ratio < 45.0:
            explanations.append(f"✅ Low Debt-to-Income ratio ({dti_ratio}%). Strong repayment buffer.")
        else:
            explanations.append(f"⚠️ High Debt-to-Income ratio ({dti_ratio}%). Elevated financial leverage.")

        if credit_score >= 700:
            explanations.append(f"✅ Excellent Credit Score ({credit_score}). Demonstrates solid credit rating.")
        elif credit_score >= 600:
            explanations.append(f"ℹ️ Average Credit Score ({credit_score}). Acceptable credit rating.")
        else:
            explanations.append(f"⚠️ Low Credit Score ({credit_score}). History indicates credit risk.")

        if payment_hist == "On-Time":
            explanations.append("✅ On-time payment history indicates responsible debt management.")
        elif payment_hist == "Delayed":
            explanations.append("⚠️ Occasional delayed payments recorded.")
        else:
            explanations.append("🔴 Past defaulted payments detected on record.")

        return {
            "prediction": prediction,
            "status": status,
            "probability": proba,
            "probability_pct": probability_pct,
            "risk_level": risk_level,
            "dti_ratio": dti_ratio,
            "explanations": explanations
        }
