import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page Setup
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="Icon",
    layout="wide"
)

# Load Model 
model = tf.keras.models.load_model("models/model.h5")

with open("models/one_hot_encoder.pkl", "rb") as file:
    one_hot_encoder = pickle.load(file)

with open("models/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# UI Header
st.title(" Customer Churn Prediction")
st.write("Estimate churn probability based on account & demographic data.")

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

# Binary field UI mapping
yes_no_mapping = {"Yes": 1, "No": 0}

geography = st.sidebar.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
credit_score = st.sidebar.number_input("Credit Score", min_value=0.0)
age = st.sidebar.slider("Age", 18, 92)
tenure = st.sidebar.slider("Tenure", 0, 10)
balance = st.sidebar.number_input("Balance", min_value=0.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4)

has_cr_card = st.sidebar.radio("Has Credit Card?", list(yes_no_mapping.keys()))
is_active_member = st.sidebar.radio("Active Member?", list(yes_no_mapping.keys()))

estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)

# Convert Yes/No to 1/0
has_cr_card_val = yes_no_mapping[has_cr_card]
is_active_member_val = yes_no_mapping[is_active_member]

# Input DataFrame
input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_val],
    'IsActiveMember': [is_active_member_val],
    'EstimatedSalary': [estimated_salary]
}

input_df = pd.DataFrame(input_data)

# One hot encode geography
geo_encoded = one_hot_encoder.transform(input_df[['Geography']]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

input_df = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)

# Scale
scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(scaled)
    proba = prediction[0][0]

    st.subheader("Prediction Outcome")
    st.write(f"###  Churn Probability: **{proba:.2f}**")

    if proba > 0.5:
        st.error(" High churn risk — customer may leave.")
    else:
        st.success(" Low churn risk — customer likely stays.")

    with st.expander("Processed Input Preview"):
        st.dataframe(input_df.style.hide(axis="index").set_properties(**{"text-align": "center"}))

