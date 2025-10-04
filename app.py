# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Gallstone Risk Tool", layout="wide")
st.title("Gallstone Risk Screening Tool")
st.markdown("### A Non-Imaging Approach to Risk Stratification")

try:
    model = joblib.load('gallstone_model.pkl')
except FileNotFoundError:
    st.error("Error: The 'gallstone_model.pkl' file was not found.")
    st.stop()
    
st.header("Patient Vitals & Lab Data")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 90, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
    with col2:
        glucose = st.number_input("Glucose", 50, 200, 100)
        tc = st.number_input("Total Cholesterol (TC)", 100, 400, 200)
        ldl = st.number_input("Low Density Lipoprotein (LDL)", 50, 250, 130)
        hdl = st.number_input("High Density Lipoprotein (HDL)", 20, 100, 40)
    with col3:
        tbw = st.number_input("Total Body Water (TBW)", 20.0, 80.0, 45.0)
        icw = st.number_input("Intracellular Water (ICW)", 10.0, 40.0, 25.0)
        ecw = st.number_input("Extracellular Water (ECW)", 10.0, 40.0, 20.0)
        vfr = st.number_input("Visceral Fat Rating (VFR)", 1, 20, 10)

st.header("Comorbidities")
col1, col2 = st.columns(2)
with col1:
    comorbidity = st.checkbox("Comorbidity")
    hypothyroidism = st.checkbox("Hypothyroidism")
    hyperlipidemia = st.checkbox("Hyperlipidemia")
with col2:
    diabetes = st.checkbox("Diabetes Mellitus (DM)")
    cad = st.checkbox("Coronary Artery Disease (CAD)")
    
if st.button("Predict Gallstone Risk"):
    gender_val = 1 if gender == "Male" else 0
    bmi = weight / ((height / 100) ** 2)
    comorbidity_val = 1 if comorbidity else 0
    hypo_val = 1 if hypothyroidism else 0
    hyper_val = 1 if hyperlipidemia else 0
    dm_val = 1 if diabetes else 0
    cad_val = 1 if cad else 0
    
    input_df = pd.DataFrame([[
        gender_val, age, comorbidity_val, cad_val, hypo_val, hyper_val, dm_val, height, weight, bmi,
        tbw, ecw, icw, 0, 0, 0, 0, vfr, 0, 0, 0, 0, 0, 0,
        0, glucose, tc, ldl, hdl, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]], columns=[
        'Gender', 'Age', 'Comorbidity', 'Coronary Artery Disease (CAD)', 'Hypothyroidism', 
        'Hyperlipidemia', 'Diabetes Mellitus (DM)', 'Height', 'Weight', 'Body Mass Index (BMI)', 
        'Total Body Water (TBW)', 'Extracellular Water (ECW)', 'Intracellular Water (ICW)', 
        'Extracellular Fluid/Total Body Water (ECF/TBW)', 'Total Body Fat Ratio (TBFR) (%)', 
        'Lean Mass (LM) (%)', 'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)', 
        'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)', 'Total Fat Content (TFC)', 
        'Visceral Fat Area (VFA)', 'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)', 
        'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 
        'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)', 
        'Alanin Aminotransferaz (ALT)', 'Alkaline Phosphatase (ALP)', 'Creatinine', 
        'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D'
    ])
    
    prediction_proba = model.predict_proba(input_df)[:, 1]
    
    if prediction_proba[0] > 0.5:
        st.success(f"Prediction: High Risk of Gallstones")
        st.write(f"Confidence: {prediction_proba[0]:.2f}")
    else:
        st.info(f"Prediction: Low Risk of Gallstones")
        st.write(f"Confidence: {1 - prediction_proba[0]:.2f}")

    st.subheader("Explanation of Prediction")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
            values=shap_values[0].values,
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ), show=False
    )
    st.pyplot(fig)