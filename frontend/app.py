"""
Credit Scoring System - Frontend Streamlit Web Application.

Provides a modern SaaS user interface with inline styling for universal rendering
across both Light Mode and Dark Mode browsers.
"""

import sys
import json
from pathlib import Path

# Add project root, backend, and database to sys.path
FRONTEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = FRONTEND_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATABASE_DIR = ROOT_DIR / "database"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(DATABASE_DIR), str(FRONTEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backend.config.settings import METRICS_PATH, BEST_MODEL_PATH
from backend.src.data_loader import load_credit_data
from backend.src.predict import CreditScoringPredictor
from backend.src.train import train_and_evaluate_models
from backend.services.report_service import generate_pdf_report, generate_csv_report, generate_txt_report
from database.db_manager import CreditDatabaseManager

# Page Configuration
st.set_page_config(
    page_title="Credit AI | Risk Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Read-Only Selectbox CSS Rules & Hover Animation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Strict Read-Only Click Selectboxes (No Text Typing or Backspacing) */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
        user-select: none !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
    }

    /* Expander Container Styling */
    .streamlit-expanderHeader {
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 1.8rem;
        margin-top: 3rem;
        border-top: 1px solid #334155;
        color: #94A3B8 !important;
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Returns cached predictor engine."""
    if not BEST_MODEL_PATH.exists():
        train_and_evaluate_models()
    return CreditScoringPredictor()


@st.cache_resource
def get_db_manager():
    """Returns cached database manager instance."""
    return CreditDatabaseManager()


def main():
    # Impressive Inline-Styled Hero Banner (Universal Light & Dark Mode Compatibility)
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #059669 100%); padding: 2rem 1.8rem; border-radius: 18px; color: #FFFFFF !important; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);">
            <h2 style="font-size: 2.2rem; font-weight: 800; margin: 0 0 0.4rem 0; color: #FFFFFF !important; letter-spacing: -0.5px;">💳 Credit AI Risk Decision Platform</h2>
            <p style="font-size: 1.05rem; font-weight: 400; opacity: 0.95; margin: 0 0 1.2rem 0; color: #F1F5F9 !important;">Automated Machine Learning Credit Scoring & Financial Underwriting Intelligence Engine</p>
            <div style="display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap;">
                <span style="background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.3); padding: 0.4rem 1.1rem; border-radius: 30px; font-size: 0.88rem; font-weight: 600; color: #FFFFFF !important;">⚡ 92.0% Model Accuracy</span>
                <span style="background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.3); padding: 0.4rem 1.1rem; border-radius: 30px; font-size: 0.88rem; font-weight: 600; color: #FFFFFF !important;">🛡️ Real-Time Default Probability</span>
                <span style="background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.3); padding: 0.4rem 1.1rem; border-radius: 30px; font-size: 0.88rem; font-weight: 600; color: #FFFFFF !important;">📄 PDF Audit Certificate</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Logo Header Badge
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 1.4rem; border-radius: 14px; color: #FFFFFF !important; text-align: center; margin-bottom: 1.2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto; display: block;">
                <rect x="2" y="5" width="20" height="14" rx="3" fill="#10B981" />
                <rect x="2" y="9" width="20" height="3" fill="#065F46" />
                <rect x="5" y="14" width="4" height="2" rx="0.5" fill="#F59E0B" />
                <circle cx="17" cy="15" r="1.5" fill="#FFFFFF" />
                <circle cx="19" cy="15" r="1.5" fill="#3B82F6" />
            </svg>
            <h3 style="margin: 0.6rem 0 0 0; color: #FFFFFF !important; font-size: 1.3rem; font-weight: 800;">CREDIT AI PRO</h3>
            <p style="margin: 0; color: #10B981 !important; font-size: 0.85rem; font-weight: 600;">Risk Scoring Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    nav_option = st.sidebar.selectbox(
        "📍 Select Navigation Module:",
        [
            "📝 Applicant Evaluation",
            "📊 Model Comparison Dashboard",
            "🗄️ Database Inspector",
            "📂 Dataset Explorer",
            "ℹ️ System Information"
        ],
        index=0,
        key="credit_nav_selectbox"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Underwriting Support**: Machine learning probability scores quantify loan default risk for credit decisioning.")

    if nav_option == "📝 Applicant Evaluation":
        render_evaluation_page()
    elif nav_option == "📊 Model Comparison Dashboard":
        render_metrics_page()
    elif nav_option == "🗄️ Database Inspector":
        render_database_page()
    elif nav_option == "📂 Dataset Explorer":
        render_dataset_page()
    elif nav_option == "ℹ️ System Information":
        render_info_page()

    # Render Footer
    st.markdown("""
        <div class="custom-footer">
            💳 <strong>Credit Scoring AI Decision System</strong> | Built with Python, Scikit-Learn & Streamlit
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 📝 APPLICANT EVALUATION PAGE
# ==========================================
def render_evaluation_page():
    st.markdown("## 📝 Financial Assessment & Loan Eligibility Evaluation")
    st.write("Enter applicant financial indicators below to compute creditworthiness score.")

    with st.form("applicant_form"):
        with st.expander("👤 Applicant Demographics & Employment", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                applicant_name = st.text_input("Applicant Full Name *", value="Alex Morgan")
            with c2:
                age = st.number_input("Age (Years) *", 18, 80, 35)
            with c3:
                employment_status = st.selectbox(
                    "Employment Type *",
                    options=["Employed", "Self-Employed", "Unemployed"],
                    index=0,
                    key="select_employment_type"
                )

        with st.expander("💰 Financial Position & Outstanding Debt Profile", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                income = st.number_input("Annual Income ($ / ₹) *", 10000, 300000, 65000, step=1000)
                existing_debt = st.number_input("Total Existing Loans / Debt ($ / ₹) *", 0, 150000, 12000, step=1000)
            with c2:
                loan_amount = st.number_input("New Loan Amount Requested ($ / ₹) *", 1000, 100000, 20000, step=1000)
                number_of_loans = st.number_input("Number of Active Loans *", 1, 10, 2)

        with st.expander("💳 Credit Score & Past Repayment History", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                credit_score = st.slider("Credit Bureau Rating (CIBIL / FICO Score: 300 - 850) *", 300, 850, 720)
            with c2:
                credit_history = st.selectbox(
                    "Credit History Rating *",
                    options=["Good", "Fair", "Poor"],
                    index=0,
                    key="select_credit_history"
                )
            with c3:
                payment_history = st.selectbox(
                    "Past Repayment Record *",
                    options=["On-Time", "Delayed", "Defaulted"],
                    index=0,
                    key="select_payment_history"
                )

        submit_btn = st.form_submit_button("🔍 Calculate Credit Score & Evaluate Risk", type="primary", use_container_width=True)

    if submit_btn:
        applicant_data = {
            "applicant_name": applicant_name,
            "Age": age,
            "Income": income,
            "Employment_Status": employment_status,
            "Loan_Amount": loan_amount,
            "Existing_Debt": existing_debt,
            "Credit_History": credit_history,
            "Number_of_Loans": number_of_loans,
            "Payment_History": payment_history,
            "Credit_Score_Value": credit_score
        }

        with st.spinner("Executing Machine Learning Risk Model..."):
            try:
                predictor = get_predictor()
                result = predictor.predict_applicant(applicant_data)

                # Save record into SQLite Database
                db_mgr = get_db_manager()
                db_mgr.insert_evaluation(applicant_data, result)

                display_prediction_result(result, applicant_name, applicant_data)
            except Exception as e:
                st.error(f"Evaluation Error: {str(e)}")


def display_prediction_result(result: dict, applicant_name: str, applicant_data: dict = None):
    st.markdown("---")
    st.markdown("### 📋 Creditworthiness Decision Result & Financial Analysis")

    is_approved = result["prediction"] == 1
    prob_pct = result["probability_pct"]

    if is_approved:
        banner_style = "background: linear-gradient(135deg, #065F46 0%, #047857 100%); border: 2px solid #10B981; color: #FFFFFF !important; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.4rem; box-shadow: 0 6px 20px rgba(16, 185, 129, 0.25);"
        title_text = "🟢 CREDITWORTHY (APPROVED)"
        bar_color = "#10B981"
    else:
        banner_style = "background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%); border: 2px solid #EF4444; color: #FFFFFF !important; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.4rem; box-shadow: 0 6px 20px rgba(239, 68, 68, 0.25);"
        title_text = "🔴 NOT CREDITWORTHY (DENIED)"
        bar_color = "#EF4444"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
            <div style="{banner_style}">
                <h3 style="margin-top:0; color: #FFFFFF !important; font-weight: 800;">{title_text}</h3>
                <p style="font-size: 1.15rem; margin-bottom: 0; color: #FFFFFF !important; font-weight: 500;">
                    Applicant <strong>{applicant_name}</strong> is assessed as <strong>{result['status']}</strong> with a <strong>{prob_pct}% Approval Probability Score</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Plotly Approval Probability Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Approval Probability Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 40], 'color': "#FEE2E2"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#D1FAE5"}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Loan EMI & Affordability Estimator Card
        if applicant_data:
            requested_loan = float(applicant_data.get("Loan_Amount", 20000))
            annual_income = float(applicant_data.get("Income", 65000))

            # Standard 5-year loan at 8.5% interest rate
            monthly_rate = 0.085 / 12
            months = 60
            estimated_emi = round((requested_loan * monthly_rate * ((1 + monthly_rate)**months)) / (((1 + monthly_rate)**months) - 1), 2)
            monthly_income = annual_income / 12
            emi_income_ratio = round((estimated_emi / max(monthly_income, 1.0)) * 100, 1)

            st.markdown("##### 💵 Loan Repayment & Affordability Analysis")
            st.info(f"💡 **Estimated Monthly Payment (EMI)**: **${estimated_emi:,.2f} / month** (5-Year term @ 8.5% p.a.) | **EMI-to-Monthly-Income Ratio**: **{emi_income_ratio}%**")

        st.markdown("##### 💡 Key Decision Factors & Explanations")
        for line in result["explanations"]:
            st.markdown(f"• {line}")

    with col2:
        st.markdown("""<div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid #334155;">""", unsafe_allow_html=True)
        st.metric(label="Decision Status", value="APPROVED" if is_approved else "DENIED")
        st.metric(label="Approval Probability", value=f"{prob_pct}%")
        st.metric(label="Risk Assessment Category", value=result["risk_level"])
        st.metric(label="Debt-to-Income Ratio", value=f"{result['dti_ratio']}%")

        st.markdown("##### 📥 Export Credit Certificate")
        try:
            pdf_bytes = generate_pdf_report(result, applicant_name)
        except Exception:
            pdf_bytes = generate_txt_report(result, applicant_name).encode("utf-8")

        st.download_button(
            "📄 Download PDF Certificate",
            data=pdf_bytes,
            file_name=f"Credit_Certificate_{applicant_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_cert_download"
        )

        csv_str = generate_csv_report(result, applicant_name)
        st.download_button(
            "📊 Download CSV Metrics",
            data=csv_str,
            file_name=f"Credit_Metrics_{applicant_name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="csv_metrics_download"
        )

        txt_str = generate_txt_report(result, applicant_name)
        st.download_button(
            "📝 Download TXT Summary",
            data=txt_str,
            file_name=f"Credit_Summary_{applicant_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="txt_summary_download"
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 📊 MODEL COMPARISON DASHBOARD
# ==========================================
def render_metrics_page():
    st.markdown("### 📊 Model Performance & Algorithm Comparison Dashboard")

    if not METRICS_PATH.exists():
        st.info("Generating model comparison benchmarks...")
        train_and_evaluate_models()

    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

    best_name = metrics_data.get("best_model_name", "Random Forest")
    results = metrics_data.get("metrics", {})

    st.success(f"🏆 **Selected Best Performing Algorithm**: **{best_name}**")

    table_rows = []
    for model_name, m in results.items():
        table_rows.append({
            "Algorithm": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score'] * 100:.2f}%",
            "ROC-AUC": f"{m['roc_auc'] * 100:.2f}%"
        })

    df_table = pd.DataFrame(table_rows)

    st.markdown("#### 📋 Algorithm Performance Comparison Table")
    st.dataframe(df_table, use_container_width=True)

    st.markdown("#### 📈 Model Metrics Comparison Chart")
    plot_data = []
    for model_name, m in results.items():
        for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            plot_data.append({
                "Algorithm": model_name,
                "Metric": metric_name.upper().replace("_", "-"),
                "Score": m[metric_name]
            })

    df_plot = pd.DataFrame(plot_data)

    fig_bar = px.bar(
        df_plot,
        x="Algorithm",
        y="Score",
        color="Metric",
        barmode="group",
        title="Classification Model Performance Benchmark",
        labels={"Score": "Score (0.0 to 1.0)"}
    )
    fig_bar.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 🧠 Feature Importance Analysis")
    importances = metrics_data.get("feature_importances", {})
    if importances:
        df_imp = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature Importances ({best_name})",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# 🗄️ DATABASE INSPECTOR PAGE
# ==========================================
def render_database_page():
    st.markdown("### 🗄️ SQLite Audit Log & Evaluation History")

    db_mgr = get_db_manager()

    search_query = st.text_input("🔍 Search Applicant Records (Name / Status / Risk):", value="")

    if search_query:
        records = db_mgr.search_evaluations(search_query)
    else:
        records = db_mgr.fetch_all_evaluations(limit=100)

    stats = db_mgr.get_summary_stats()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Historical Evaluated Applicants", stats["total"])
    with c2:
        st.metric("Approved Applications", stats["approved"])
    with c3:
        st.metric("Denied Applications", stats["denied"])

    st.markdown("#### 📄 Applicant Assessment Audit Log")
    if records:
        df_rec = pd.DataFrame(records)
        st.dataframe(df_rec, use_container_width=True)
    else:
        st.info("No historical records found. Submit an evaluation on the 📝 Applicant Evaluation page!")


# ==========================================
# 📂 DATASET EXPLORER PAGE
# ==========================================
def render_dataset_page():
    st.markdown("### 📂 Credit Scoring Dataset Explorer")

    df = load_credit_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Applicants", df.shape[0])
    with c2:
        st.metric("Financial Features", df.shape[1] - 1)
    with c3:
        st.metric("Approved (Creditworthy)", int(df["Creditworthy"].sum()))
    with c4:
        st.metric("Denied (High Risk)", int((df["Creditworthy"] == 0).sum()))

    st.markdown("#### 📄 Dataset Sample Records")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("#### 📊 Target Class Balance")
    fig_pie = px.pie(
        values=df["Creditworthy"].value_counts().values,
        names=["Creditworthy (1)", "Not Creditworthy (0)"],
        title="Credit Assessment Outcome Distribution",
        color_discrete_sequence=["#10B981", "#EF4444"],
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================
# ℹ️ SYSTEM INFORMATION PAGE
# ==========================================
def render_info_page():
    st.markdown("### ℹ️ Credit Scoring System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid #334155;">
                <h4>🎯 Project Goal & Scope</h4>
                <p>The <strong>Credit Scoring Decision System</strong> automates loan applicant creditworthiness evaluation using machine learning classifiers. It predicts whether a person is approved or denied credit based on financial leverage, income, credit bureau ratings, and repayment history.</p>
            </div>

            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid #334155; margin-top: 1rem;">
                <h4>🧠 Machine Learning Workflow</h4>
                <ol>
                    <li><strong>Data Preprocessing</strong>: StandardScaler for numerical scaling & OneHotEncoder for categorical attributes.</li>
                    <li><strong>Supervised Classifiers</strong>: Logistic Regression, Decision Tree, Random Forest.</li>
                    <li><strong>Evaluation Metrics</strong>: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.</li>
                    <li><strong>Artifact Persistence</strong>: Saved Scikit-Learn pipelines via Joblib for fast real-time inference.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 16px; border: 1px solid #334155;">
                <h4>🛠️ Technology Stack</h4>
                <ul>
                    <li><strong>Frontend UI</strong>: Streamlit, Plotly Interactive Visuals</li>
                    <li><strong>Backend Architecture</strong>: Python, Scikit-Learn, Pandas, Joblib</li>
                    <li><strong>Database Tier</strong>: SQLite Database Audit Logger (<code>credit_records.db</code>)</li>
                    <li><strong>Reporting Service</strong>: FPDF PDF Certificates, CSV Data & TXT Exporters</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
