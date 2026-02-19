import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")

# ----------------------------
# Load Model Files
# ----------------------------
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f8fb;
    }
    h1 {
        color: #0e4d92;
        text-align: center;
    }
    .stButton>button {
    background: linear-gradient(135deg, #2563EB, #1E40AF);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.title("🧠 Stroke Risk Prediction System")
st.write("Enter patient health details below to estimate stroke risk probability.")

st.divider()

# ----------------------------
# Input Section (Organized in columns)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 50)
    avg_glucose = st.slider("Average Glucose Level", 50, 300, 120)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)

with col2:
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    gender = st.selectbox("Gender", ["Male", "Female"])

ever_married = st.selectbox("Ever Married", ["Yes", "No"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

st.divider()

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Stroke Risk"):

    input_dict = {
        "age": age,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender_Male": 1 if gender == "Male" else 0,
        "ever_married_Yes": 1 if ever_married == "Yes" else 0,
        "Residence_type_Urban": 1 if residence == "Urban" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_Govt_job": 1 if work_type == "Govt_job" else 0,
        "work_type_children": 1 if work_type == "children" else 0,
        "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
        "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
        "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
        "smoking_status_Unknown": 1 if smoking_status == "Unknown" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Custom healthcare threshold
    threshold = 0.35
    high_risk = probability >= threshold

    st.subheader("Prediction Result")

    st.progress(int(probability * 100))

    if high_risk:
        st.error(f"""
        ⚠️ HIGH STROKE RISK DETECTED  

        Risk Probability: {probability:.2f}  
        Threshold Used: {threshold}
        """)
    else:
        st.success(f"""
        ✅ LOW STROKE RISK  

        Risk Probability: {probability:.2f}  
        Threshold Used: {threshold}
        """)
