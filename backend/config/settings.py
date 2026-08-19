"""
Backend Configuration Settings for Credit Scoring System.
"""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
DATABASE_DIR = ROOT_DIR / "database"

DATA_DIR = BACKEND_DIR / "dataset"
MODELS_DIR = BACKEND_DIR / "models"
SERVICES_DIR = BACKEND_DIR / "services"

DATASET_PATH = DATA_DIR / "credit_scoring.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
