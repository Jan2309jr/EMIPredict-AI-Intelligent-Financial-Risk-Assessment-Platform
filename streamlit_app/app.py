# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import traceback

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="💰 EMIPredict AI", layout="centered")
st.title("💰 EMIPredict AI – Intelligent Financial Risk Assessment")
st.markdown("""
Predict EMI eligibility and maximum EMI limit using your financial data.  
_Powered by AI models trained on real credit datasets._
""")

# -------------------------
# Paths (works when running from streamlit_app/)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "emi_prediction_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}")
        st.stop()
    df = pd.read_csv(path)
    return df

df = load_dataset()
st.success(f"✅ Dataset loaded successfully from `{os.path.relpath(DATA_PATH)}`")
st.dataframe(df.head())

# -------------------------
# Helper: preprocess & optional training
# -------------------------
def build_preprocessor(X: pd.DataFrame):
    # numeric / categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Ensure categorical columns are strings to avoid mixed-type issues
    # This does not modify the original df passed in by value when used correctly.
    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].astype(str).fillna("missing")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder param name differs across sklearn releases
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ], sparse_threshold=0)

    return preprocessor, numeric_cols, categorical_cols

def prepare_training_data(df, target_clf="emi_eligibility", target_reg="max_monthly_emi", sample_size=20000, random_state=42):
    df = df.copy()
    df = df[df[target_clf].notna() & df[target_reg].notna()]

    if df.shape[0] > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    X = df.drop(columns=[target_clf, target_reg])
    # identify categorical columns robustly (also include object-like numeric columns)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    # Coerce categorical columns to string and fill missing
    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].astype(str).fillna("missing")

    # For any remaining non-numeric columns, coerce as string too
    non_numeric = [c for c in X.columns if X[c].dtype == "O" or X[c].dtype == "object"]
    if non_numeric:
        X[non_numeric] = X[non_numeric].astype(str).fillna("missing")

    y_clf = df[target_clf].astype(str).str.strip()
    y_reg = df[target_reg].astype(float)
    return X, y_clf, y_reg
# -------------------------
# Train lightweight models if missing
# -------------------------
def train_and_save_models(df):
    st.info("Training lightweight models because saved models are missing. This may take a short time.")
    X, y_clf, y_reg = prepare_training_data(df)

    # build preprocessor from X
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # encode classification target
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    # train-test split
    X_train, X_val, yclf_train, yclf_val, yreg_train, yreg_val = train_test_split(
        X, y_clf_enc, y_reg, test_size=0.2, random_state=42, stratify=y_clf_enc
    )

    # build pipelines
    clf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42))
    ])
    reg_pipeline = Pipeline([
        ("pre", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42))
    ])

    # fit
    clf_pipeline.fit(X_train, yclf_train)
    reg_pipeline.fit(X_train, yreg_train)

    # Save pipelines and label encoder
    joblib.dump(clf_pipeline, os.path.join(MODELS_DIR, "best_classifier.joblib"))
    joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "best_regressor.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    # Also save standalone encoder/scaler if user expects them (extract from preprocessor)
    try:
        # Save a copy of the fitted preprocessor so app can reuse it if needed
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    except Exception:
        pass

    st.success("✅ Training completed and models saved to `models/`")
    return clf_pipeline, reg_pipeline, le

# -------------------------
# Load or train models
# -------------------------
@st.cache_resource
def get_models():
    clf_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    reg_path = os.path.join(MODELS_DIR, "best_regressor.joblib")
    label_enc_path = os.path.join(MODELS_DIR, "label_encoder.joblib")

    missing = []
    if not os.path.exists(clf_path):
        missing.append(clf_path)
    if not os.path.exists(reg_path):
        missing.append(reg_path)
    if not os.path.exists(label_enc_path):
        # label encoder might be inside models but not required to exist separately
        missing.append(label_enc_path)

    if missing:
        # train lightweight models on dataset and save them
        clf, reg, le = train_and_save_models(df)
        return clf, reg, le

    # else load
    try:
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        le = joblib.load(label_enc_path)
        return clf, reg, le
    except Exception as e:
        st.error("Error loading model files. Will attempt to retrain.")
        st.text(traceback.format_exc())
        clf, reg, le = train_and_save_models(df)
        return clf, reg, le

clf, reg, label_encoder = get_models()
st.success("✅ Models ready.")

# -------------------------
# User input UI
# -------------------------
st.header("🧮 EMI Eligibility Prediction")

col1, col2 = st.columns(2)
with col1:
    salary = st.number_input("Monthly Salary (₹)", min_value=1000, value=50000, step=1000)
    current_emi = st.number_input("Current EMI (₹)", min_value=0, value=0, step=500)
    expenses = st.number_input("Other Expenses (₹)", min_value=0, value=2000, step=500)
    years = st.number_input("Years of Employment", min_value=0, value=5, step=1)
    travel = st.number_input("Travel Expenses (₹)", min_value=0, value=1000, step=500)
    groceries = st.number_input("Groceries & Utilities (₹)", min_value=0, value=5000, step=500)

with col2:
    rent = st.number_input("Monthly Rent (₹)", min_value=0, value=8000, step=500)
    loan = st.number_input("Existing Loans (₹)", min_value=0, value=0, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750, step=10)
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=50000, value=500000, step=50000)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, value=60, step=6)
    age = st.number_input("Age", min_value=18, value=30, step=1)

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col4:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
with col5:
    education = st.selectbox("Education", ["Graduate", "Postgraduate", "HighSchool"])

col6, col7, col8 = st.columns(3)
with col6:
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
with col7:
    company_type = st.selectbox("Company Type", ["Private", "Public", "Startup", "Other"])
with col8:
    house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided"])

emi_scenario = st.selectbox("EMI Scenario", ["Standard", "Flexible", "High Risk"])

# -------------------------
# Predict
# -------------------------
if st.button("🔍 Predict EMI Eligibility & Limit"):
    try:
        df_raw = pd.DataFrame([{
            "age": age,
            "monthly_salary": salary,
            "years_of_employment": years,
            "monthly_rent": rent,
            "family_size": 3,
            "dependents": 1,
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": travel,
            "groceries_utilities": groceries,
            "other_monthly_expenses": expenses,
            "existing_loans": loan,
            "current_emi_amount": current_emi,
            "credit_score": credit_score,
            "bank_balance": 50000,
            "emergency_fund": 10000,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "employment_type": employment_type,
            "company_type": company_type,
            "house_type": house_type,
            "emi_scenario": emi_scenario
        }])

        st.write("### 🧾 Input Data Sent to Model")
        st.dataframe(df_raw)

        # predict classification
        class_pred_raw = clf.predict(df_raw)
        # if label encoder exists, convert back; label_encoder may be sklearn LabelEncoder object
        try:
            # if classifier pipeline predicts encoded ints
            if isinstance(class_pred_raw[0], (np.integer, int)):
                label = label_encoder.inverse_transform([int(class_pred_raw[0])])[0]
            else:
                # if classifier pipeline returns string label directly
                label = str(class_pred_raw[0])
        except Exception:
            # fallback
            label = str(class_pred_raw[0])

        st.markdown("### 🏦 **Predicted EMI Eligibility**")
        if hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(df_raw)[0]
                # choose highest-prob index
                idx = np.argmax(proba)
                confidence = float(proba[idx])
                st.success(f"{label} ({confidence*100:.1f}% confidence)")
                st.progress(confidence)
            except Exception:
                st.success(label)
        else:
            st.success(label)

        # predict regression (max monthly emi)
        emi_pred_raw = reg.predict(df_raw)
        emi_val = float(emi_pred_raw[0])
        st.markdown("### 💸 **Predicted Maximum Affordable EMI**")
        st.info(f"₹{emi_val:,.0f}")

        st.divider()
        colA, colB = st.columns(2)
        colA.metric("Eligibility", label)
        colB.metric("Predicted EMI", f"₹{emi_val:,.0f}")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)
        st.text(traceback.format_exc())

# -------------------------
# Debug Info
# -------------------------
with st.expander("🧩 Model & Preprocessing Info"):
    try:
        st.write("Models directory:", os.path.relpath(MODELS_DIR))
        for fname in sorted(os.listdir(MODELS_DIR)):
            st.write("-", fname)
    except Exception as e:
        st.exception(e)
