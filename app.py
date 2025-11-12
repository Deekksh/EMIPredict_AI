# app.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import joblib
import os
from io import StringIO

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="EMIPredict AI", layout="wide")
MLFLOW_CLASS_MODEL = os.getenv("MLFLOW_CLASS_MODEL", "models:/XGBoost_EMI_Classifier/1")
MLFLOW_REG_MODEL = os.getenv("MLFLOW_REG_MODEL", "models:/XGBoost_EMI_Amount_Predictor/1")
LOCAL_CLASS_PATH = os.getenv("LOCAL_CLASS_PATH", "artifacts/best_classifier.joblib")
LOCAL_REG_PATH = os.getenv("LOCAL_REG_PATH", "artifacts/best_regressor.joblib")

# ---------------------------
# Utilities: load models (MLflow first, fallback to local)
# ---------------------------
@st.cache_resource
def load_ml_models():
    """Try to load MLflow models; if fail, load local joblib files."""
    clf, reg = None, None
    # classifier
    try:
        clf = mlflow.pyfunc.load_model(MLFLOW_CLASS_MODEL)
    except Exception as e:
        st.warning(f"MLflow classifier load failed: {e} — attempting local file.")
        if os.path.exists(LOCAL_CLASS_PATH):
            clf = joblib.load(LOCAL_CLASS_PATH)
        else:
            st.error("No classifier model available (MLflow + local fallback failed).")
    # regressor
    try:
        reg = mlflow.pyfunc.load_model(MLFLOW_REG_MODEL)
    except Exception as e:
        st.warning(f"MLflow regressor load failed: {e} — attempting local file.")
        if os.path.exists(LOCAL_REG_PATH):
            reg = joblib.load(LOCAL_REG_PATH)
        else:
            st.error("No regression model available (MLflow + local fallback failed).")
    return clf, reg

clf_model, reg_model = load_ml_models()

# ---------------------------
# Helper: preprocess single-row input to model features
# (Adapt this to the exact preprocessing you used in training)
# ---------------------------
def build_input_df(raw):
    """
    raw: dict of input fields.
    Return: DataFrame with same columns/order expected by the models.
    IMPORTANT: ensure this matches your training preprocessing & feature engineering.
    """
    # Example minimal mapping: extend to match your feature pipeline
    df = pd.DataFrame([raw])
    # Example conversions (update names & engineered features used in training):
    # df['some_ratio'] = df['current_emi_amount'] / (df['monthly_salary'] + 1)
    # ... add the same feature-engineering steps you applied during training
    return df

# ---------------------------
# Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction", "Model Info", "Admin"])

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.header("EMIPredict AI — Real-time EMI Eligibility & Amount Prediction")
    st.markdown("""
    **How to use**
    - Single Prediction: fill the form and press Predict
    - Batch Prediction: upload a CSV containing columns matching training features
    - Model Info: view model name & quick summary (MLflow or local)
    - Admin: simple CRUD (local CSV) for demo/admin operations
    """)
    st.divider()
    st.write("Models loaded status:")
    st.write(f"Classifier: {'loaded' if clf_model is not None else 'not loaded'}")
    st.write(f"Regressor: {'loaded' if reg_model is not None else 'not loaded'}")
    st.info("Ensure feature preprocessing in `build_input_df()` mirrors training preprocessing exactly.")

# ---------------------------
# Single Prediction Page
# ---------------------------
elif page == "Single Prediction":
    st.header("Single Record Prediction")
    # Example input fields (replace/add with your full 22 features)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000, step=1000)
    with col2:
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    with col3:
        current_emi_amount = st.number_input("Current EMI Amount", min_value=0, value=0, step=100)
        bank_balance = st.number_input("Bank Balance", min_value=0, value=50000, step=1000)
        emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])

    raw = {
        "age": age,
        "gender": gender,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "credit_score": credit_score,
        "current_emi_amount": current_emi_amount,
        "bank_balance": bank_balance,
        "emi_scenario": emi_scenario,
        # add other required fields...
    }

    st.subheader("Preview input")
    st.json(raw)

    if st.button("Predict"):
        if clf_model is None or reg_model is None:
            st.error("Models not loaded. Check logs or environment variables.")
        else:
            X_in = build_input_df(raw)
            # MLflow pyfunc models expect pandas DataFrame and return numpy arrays/DF
            try:
                cls_pred = clf_model.predict(X_in)
                # For pyfunc models this might return string class; for SKLearn-persisted joblib it returns numeric
                reg_pred = reg_model.predict(X_in)
            except Exception as e:
                st.exception(f"Prediction failed: {e}")
                cls_pred, reg_pred = None, None

            st.markdown("### Results")
            st.write("Classification (EMI eligibility):", cls_pred[0] if cls_pred is not None else "N/A")
            st.write("Regression (Predicted max monthly EMI):", f"₹ {reg_pred[0]:,.2f}" if reg_pred is not None else "N/A")

# ---------------------------
# Batch Prediction Page
# ---------------------------
elif page == "Batch Prediction":
    st.header("Batch Prediction (CSV upload)")
    st.markdown("Upload a CSV with the same columns used in training (raw features).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df_in.head(5))

        if st.button("Run Batch Prediction"):
            if clf_model is None or reg_model is None:
                st.error("Models not loaded. Cannot run batch.")
            else:
                X_batch = df_in.copy()  # apply preprocessing if required
                try:
                    cls_preds = clf_model.predict(X_batch)
                    reg_preds = reg_model.predict(X_batch)
                    out = df_in.copy()
                    out["predicted_emi_eligibility"] = cls_preds
                    out["predicted_max_monthly_emi"] = reg_preds
                    st.success("Batch prediction complete.")
                    st.dataframe(out.head(50))
                    csv = out.to_csv(index=False)
                    st.download_button("Download predictions CSV", csv, file_name="predictions.csv")
                except Exception as e:
                    st.exception(f"Batch prediction error: {e}")

# ---------------------------
# Model Info Page
# ---------------------------
elif page == "Model Info":
    st.header("Model Information")
    st.write("Classifier model URI:", MLFLOW_CLASS_MODEL)
    st.write("Regressor model URI:", MLFLOW_REG_MODEL)
    st.divider()
    st.subheader("Notes")
    st.write("""
    - Models are loaded from MLflow model registry URI if available; otherwise local joblib is used.
    - Ensure the feature order and preprocessing match the training pipeline exactly.
    """)

# ---------------------------
# Admin Page (Lightweight CRUD demo using CSV)
# ---------------------------
elif page == "Admin":
    st.header("Admin / Data Management (Demo)")
    st.markdown("This is a demo CRUD using a CSV file in the repo. For production use a DB (Postgres) + auth.")
    DATAFILE = "data/emi_admin_db.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATAFILE):
        pd.DataFrame(columns=["customer_id", "age", "monthly_salary", "emi_eligibility"]).to_csv(DATAFILE, index=False)

    df_admin = pd.read_csv(DATAFILE)
    st.write("Current records:", df_admin.shape[0])
    st.dataframe(df_admin.head(10))

    st.subheader("Add new record")
    cid = st.text_input("customer_id")
    a = st.number_input("age", min_value=18, max_value=100, value=30)
    sal = st.number_input("monthly_salary", value=50000)
    elig = st.selectbox("emi_eligibility", ["Eligible", "High_Risk", "Not_Eligible"])
    if st.button("Add record"):
        new = {"customer_id": cid, "age": a, "monthly_salary": sal, "emi_eligibility": elig}
        df_admin = pd.concat([df_admin, pd.DataFrame([new])], ignore_index=True)
        df_admin.to_csv(DATAFILE, index=False)
        st.success("Record added.")
        st.experimental_rerun()
