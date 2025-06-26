import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and feature names
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_name = pickle.load(open('feature_name.pkl', 'rb'))

st.title("Heart Disease Prediction App")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure")
cholesterol = st.number_input("Cholesterol")
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate")
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak")
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encode manually (one-hot encoded like training)
input_dict = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'ExerciseAngina_Y': 1 if exercise_angina == "Yes" else 0,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == "Male" else 0,
    'ChestPainType_ASY': 1 if chest_pain == "ASY" else 0,
    'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
    'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
    'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
    'RestingECG_LVH': 1 if resting_ecg == "LVH" else 0,
    'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
    'ST_Slope_Up': 1 if st_slope == "Up" else 0,
}

# Fill in 0 for any missing columns
for col in feature_name:
    if col not in input_dict:
        input_dict[col] = 0

# Create DataFrame in correct order
input_df = pd.DataFrame([input_dict])[feature_name]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "No Heart Disease Detected" if prediction == 0 else "⚠️ High Risk of Heart Disease"
    st.success(result)
