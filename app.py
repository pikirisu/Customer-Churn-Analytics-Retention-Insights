import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


st.set_page_config(
    page_title="Churn Predictor",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0b0f19;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #2d3748;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
    margin-top: 15px;
}

.metric-box {
    background: #161b22;
    padding: 15px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #2d333b;
}

.metric-title {
    font-size: 14px;
    color: #9ca3af;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: white;
}

.risk-high {
    color: #ff4b4b;
    font-size: 20px;
    font-weight: 700;
}

.risk-mid {
    color: #facc15;
    font-size: 20px;
    font-weight: 700;
}

.risk-low {
    color: #00ff99;
    font-size: 20px;
    font-weight: 700;
}

.stButton button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(to right, #2563eb, #06b6d4);
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/model.h5")

    with open("models/one_hot_encoder.pkl", "rb") as f:
        one_hot_encoder = pickle.load(f)

    with open("models/label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, one_hot_encoder, label_encoder_gender, scaler


model, one_hot_encoder, label_encoder_gender, scaler = load_artifacts()

st.markdown("##  Churn Predictor")
st.markdown(
    "### Predict customer churn risk using machine learning & banking behavior insights"
)

st.divider()

left, right = st.columns([1.1, 2])

yes_no_mapping = {"Yes": 1, "No": 0}


with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(" Customer Inputs")

    st.markdown("####  Demographics")
    geography = st.selectbox("Region", one_hot_encoder.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 35)

    st.markdown("####  Financial Profile")
    credit_score = st.slider("Credit Score", 300, 900, 650)
    balance = st.number_input("Account Balance ($)", value=50000.0)
    salary = st.number_input("Estimated Salary ($)", value=60000.0)

    st.markdown("####  Engagement")
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    num_products = st.slider("Products Used", 1, 4, 2)

    has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    active_member = st.radio("Active Member?", ["Yes", "No"], horizontal=True)

    predict_btn = st.button(" Run Churn Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader(" Risk Analytics Dashboard")

    if predict_btn:

        has_card_val = yes_no_mapping[has_card]
        active_val = yes_no_mapping[active_member]

        input_data = {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card_val],
            "IsActiveMember": [active_val],
            "EstimatedSalary": [salary],
        }

        input_df = pd.DataFrame(input_data)

        geo_encoded = one_hot_encoder.transform(input_df[["Geography"]]).toarray()
        geo_df = pd.DataFrame(
            geo_encoded,
            columns=one_hot_encoder.get_feature_names_out(["Geography"])
        )

        input_df = pd.concat([input_df.drop("Geography", axis=1), geo_df], axis=1)

        scaled_input = scaler.transform(input_df)

        with st.spinner("Running churn model..."):
            proba = float(model.predict(scaled_input)[0][0])


        k1, k2, k3, k4 = st.columns(4)

        k1.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>Credit Score</div>
            <div class='metric-value'>{credit_score}</div>
        </div>
        """, unsafe_allow_html=True)

        k2.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>Balance</div>
            <div class='metric-value'>${balance:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        k3.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>Salary</div>
            <div class='metric-value'>${salary:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        risk_label = "LOW"
        risk_class = "risk-low"

        if proba > 0.7:
            risk_label = "HIGH"
            risk_class = "risk-high"
        elif proba > 0.4:
            risk_label = "MEDIUM"
            risk_class = "risk-mid"

        k4.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>Risk Level</div>
            <div class='metric-value {risk_class}'>{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.metric("Churn Probability", f"{proba:.2%}")
        st.progress(int(proba * 100))

        st.markdown(f"### Risk Assessment: <span class='{risk_class}'>{risk_label}</span>",
                    unsafe_allow_html=True)

        st.markdown("####  Model Insight")
        if credit_score < 500:
            st.write("• Low credit score increases churn likelihood.")
        if active_val == 0:
            st.write("• Inactive customers churn more frequently.")
        if balance > 100000:
            st.write("• High-balance customers may switch banks for better offers.")

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander(" View Model Features"):
            st.dataframe(input_df)

    else:
        st.info("Fill customer details and click **Run Churn Prediction** to view risk analytics.")
