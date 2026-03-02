import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest.pkl")

st.title("🚀 Predictive Lead Intelligence")
st.subheader("Lead Conversion Prediction App")

st.write("Enter customer details below:")

# User Inputs
age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Account Balance", value=1000)
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact]
})

# One-hot encode like training
input_data = pd.get_dummies(input_data)

# Align columns with training data
model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict Conversion"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"### Conversion Probability: {probability:.2%}")

    if prediction == 1:
        st.success("High likelihood of conversion 🎯")
    else:
        st.warning("Low likelihood of conversion ❌")