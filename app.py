# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')

st.set_page_config(page_title="Customer Retention Tool", page_icon="📊")

st.title("📊 Customer Retention Predictor")
st.write("This AI tool predicts if a customer is at risk of leaving (churning) based on their contract details.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Months with Company (Tenure)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)

with col2:
    contract = st.selectbox("Contract Type", encoders['Contract'].classes_)
    payment = st.selectbox("Payment Method", encoders['PaymentMethod'].classes_)

# Auto-calculate Total Charges (approximate)
total_charges = tenure * monthly_charges
st.info(f"Estimated Total Charges: ${total_charges:.2f}")

# --- PREDICTION LOGIC ---
if st.button("Predict Churn Risk"):
    # 1. Encode the inputs using the saved encoders
    contract_encoded = encoders['Contract'].transform([contract])[0]
    payment_encoded = encoders['PaymentMethod'].transform([payment])[0]

    # 2. Prepare data for model
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract_encoded],
        'PaymentMethod': [payment_encoded]
    })

    # 3. Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # 4. Display Result
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk! This customer has a {probability:.0%} chance of leaving.")
        st.write("**Recommended Action:** Offer a 10% discount on the 1-year contract.")
    else:
        st.success(f"✅ Safe. This customer has only a {probability:.0%} chance of leaving.")