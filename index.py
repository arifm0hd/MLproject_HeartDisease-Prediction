import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Ada masa for Heart Attack?",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/arifm0hd',
    }
)

st.title("Heart Attack Predictor 🫀")

# function if the user choose KNN model
def knn_display(option):
    st.write("KNN has the accuracy of 0.785")
    df_pred = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    knn_cld_model = joblib.load("knn_clf.pkl")
    prediction = knn_cld_model.predict(df_pred)

    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You should seek the doctor, potential heart problem")
        else:
            st.write("You're fine insyaAllah")
   
# function if the user choose SVM model
def svm_display(option):
    st.write("SVM has the accuracy of 0.839")
    df_pred = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    svm_clf_model = joblib.load("svm_clf.pkl")
    prediction = svm_clf_model.predict(df_pred)

    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You should seek the doctor, potential heart problem")
        else:
            st.write("You're fine insyaAllah")

# function if the user choose RF model
def rf_display(option):
    st.write("RF has the accuracy of 0.937")
    df_pred = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    rf_clf_model = joblib.load("rf_clf.pkl")
    prediction = rf_clf_model.predict(df_pred)

    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You should seek the doctor, potential heart problem")
        else:
            st.write("You're fine insyaAllah")

# function if the user choose GB model
def gb_display(option):
    st.write("GB has the accuracy of 0.941")
    gb_clf_model = joblib.load("gb_clf.pkl")
    df_pred = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = gb_clf_model.predict(df_pred)

    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You should seek the doctor, potential heart problem")
        else:
            st.write("You're fine insyaAllah")

# function if the user choose RF-GB model
def hybrid_display(option):
    st.write("RF-GB Hybrid has the accuracy of 0.951")
    hybrid_clf_model = joblib.load("hybrid_clf.pkl")
    df_pred = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = hybrid_clf_model.predict(df_pred)

    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You should seek the doctor, potential heart problem")
        else:
            st.write("You're fine insyaAllah")

st.divider()

# User inputs
age = st.slider("Age",29,77)
sex = st.radio("Gender (0-Female, 1-Male)", (0, 1))
cp = st.radio("Chest Pain Type (Typical Angina (TA), Atypical Angina (ATA), Non-Anginal Pain (NPA), and Asymptotic",(0, 1, 2, 3))
trestbps = st.slider("Resting Blood Pressure (mmHg)",90,200)
chol = st.slider("Cholesterol Level (mg/dl)",125,565)
fbs = st.radio("Fasting Blood Sugar > 120mg/dl (0-No, 1-Yes)", (0, 1))
restecg = st.radio("Resting ECG Class (0-Normal, 1-ST, 2-LVH)", (0, 1, 2))
thalach = st.slider("Max heart rate (Stress Test)",70,205)
exang = st.radio("Exercise Induced Angina (0-No, 1-Yes)", (0, 1))
oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest",0.0,6.2)
slope = st.radio("Slope of the Peak Exercise ST Segment (0-Upsloping, 1-Flat, 2-Downsloping)", (0, 1, 2))
ca = st.radio("Number of Major Vessels (0-3) Colored by Fluoroscopy", (0, 1, 2, 3))
thal = st.radio("Thalassemia (1-Normal, 2-Fixed Defect, 3-Reversible Defect)", (1, 2, 3))

st.divider()

option = st.selectbox("Which model do you want to use?", ("K-Nearest Neighbor (KNN)","Support Vector Machine (SVM)","Random Forest (RF)","Gradient Boosting (GB)","RF-GB Hybrid"))

if option=="K-Nearest Neighbor (KNN)":
    knn_display(option)
elif option=="Support Vector Machine (SVM)":
    svm_display(option)
elif option=="Random Forest (RF)":
    rf_display(option)
elif option=="Gradient Boosting (GB)":
    gb_display(option)
elif option=="Random Forest + Gradient Boosting Hybrid (RF-GB)":
    hybrid_display(option)
