# 💳 Decoupled Credit Scoring & Risk Evaluation System

An enterprise-grade Python & Machine Learning credit scoring platform with **3 decoupled top-level directories**: `frontend/`, `backend/`, and `database/`.

Predicts whether a loan applicant is **Creditworthy (Approved)** or **Not Creditworthy (Denied)** based on financial profile, debt leverage, credit bureau rating, and payment history.

---

## 📂 3-Tier Decoupled Project Architecture

```
koushik_credit_scoring/
├── frontend/                        # STREAMLIT UI TIER
│   └── app.py                       # Modern web dashboard & visual controls
├── backend/                         # MACHINE LEARNING & SERVICES TIER
│   ├── config/                      # Backend settings & paths
│   │   └── settings.py
│   ├── dataset/                     # Financial datasets
│   │   └── credit_scoring.csv
│   ├── models/                      # ML model artifacts
│   │   ├── best_model.joblib
│   │   ├── preprocessor.joblib
│   │   └── model_metrics.json
│   ├── services/                    # Reporting & certificate export services
│   │   └── report_service.py
│   └── src/                         # ML Pipeline source code
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── train.py
│       └── predict.py
├── database/                        # SQLITE DATABASE AUDIT TIER
│   ├── db_manager.py                # Database Manager & Audit Inspector
│   └── credit_records.db            # SQLite database file
├── app.py                           # Root VS Code entrypoint
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation & setup guide
```

---

## 🚀 Quick Start Guide (VS Code)

### 1. Open Project Folder in VS Code

Click **File** ➡️ **Open Folder...** ➡️ select `/Users/koushik/Desktop/koushik_credit_scoring`.

### 2. Open Terminal & Activate Environment

Open terminal (`Cmd + ~`) and run:

```bash
source venv/bin/activate
```

### 3. Launch Web Application

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser!

---

## 📊 Evaluated Classification Models & Leaderboard

- **Logistic Regression**: Accuracy 92.0%, F1-Score 0.9375, ROC-AUC 0.9723 *(Top Model)*
- **Random Forest**: Accuracy 85.0%, F1-Score 0.8889
- **Decision Tree**: Accuracy 79.0%, F1-Score 0.8409
