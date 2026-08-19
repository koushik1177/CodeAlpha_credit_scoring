"""
Model Training and Comparison Engine for Credit Scoring System.

Trains, evaluates, and compares Logistic Regression, Decision Tree, and Random Forest
classifiers. Saves the best model artifact, preprocessor pipeline, and evaluation metrics.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from backend.config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, MODELS_DIR
from backend.src.data_loader import load_credit_data
from backend.src.preprocessing import build_preprocessor_pipeline, save_preprocessor, TARGET_COLUMN


def train_and_evaluate_models() -> dict:
    """
    Trains Logistic Regression, Decision Tree, and Random Forest models,
    evaluates performance, selects the top model, and persists model artifacts.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = load_credit_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Fit Preprocessing Pipeline
    preprocessor = build_preprocessor_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save Preprocessor Pipeline
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)

    # 4. Define Classifier Suite
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8)
    }

    results = {}
    best_f1 = -1.0
    best_model_name = ""
    best_model_obj = None

    print("\n==================================================")
    print("      MODEL TRAINING & EVALUATION LEADERBOARD     ")
    print("==================================================")

    for name, model in models.items():
        # Fit Model
        model.fit(X_train_trans, y_train)

        # Predictions & Probabilities
        y_pred = model.predict(X_test_trans)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_trans)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # Calculate Evaluation Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm
        }

        print(f"Algorithm: {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        # Best Model Selection Based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print("--------------------------------------------------")
    print(f"🏆 Best Performing Model: {best_model_name} (F1: {best_f1:.4f})")
    print("==================================================\n")

    # 5. Save Best Model Artifact
    joblib.dump(best_model_obj, BEST_MODEL_PATH)

    # Get Feature Importances if available
    feature_importances = {}
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"Feature_{i}" for i in range(X_train_trans.shape[1])]

    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
        feature_importances = dict(zip(feature_names, [round(float(val), 4) for val in importances]))
    elif hasattr(best_model_obj, "coef_"):
        coefs = np.abs(best_model_obj.coef_[0])
        feature_importances = dict(zip(feature_names, [round(float(val), 4) for val in coefs]))

    # Save Evaluation Metrics Summary JSON
    summary_data = {
        "best_model_name": best_model_name,
        "metrics": results,
        "feature_importances": feature_importances
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary_data, f, indent=4)

    return summary_data


if __name__ == "__main__":
    train_and_evaluate_models()
