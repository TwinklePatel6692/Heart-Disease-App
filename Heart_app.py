import streamlit as st
import pandas as pd
import joblib   #  for use unpickle

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns =joblib.load("columns.pkl")

st.title("Heart Disease Prediction by Twinkle ❤️") 
st.markdown(" Please Provide the following Details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M","F"])
Chest_Pain =st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure(mm hg)",80,200,120)
Cholesterol = st.number_input("Cholesterol(mg/dL)", 100, 600, 200)
FastingBS = st.selectbox("Fsting Blood Suger > 120 mg/dL", [0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal", "ST", "LVH"])
max_HR = st.slider("Max Heart Rate ", 60, 220, 150, )
exercise_angina = st.selectbox("Exercise-Induced Angina ", ["Y","N"])
oldpeak = st.slider("OldPeak (ST depression)",0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["UP", "Flat", "Down"])


if st.button("Predict"):
    raw_input = {
        "Age" : age,
        "Sex_" + sex :1,
        "RestinBP" : resting_bp,
        "Cholesterol" : Cholesterol,
        "FastingBS" : FastingBS,
        "Max HR" : max_HR,
        "Oldpeak" : oldpeak,
        "ST_Slope_" + st_slope : 1,
        "RestingECG_" + resting_ecg : 1,
        "ExerciseAngina_" + exercise_angina : 1,
        "ChestPainType_" + Chest_Pain : 1,

    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]        

    scaled_input = scaler.transform(input_df)
    Prediction = model.predict(scaled_input)[0]

    if Prediction == 1 :
        st.error("⚠️ High Risk of Heart Disease")
    else :
        st.success("✅ Low Risk of Heart Disease")    