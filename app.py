import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Predictor", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    💰 Loan Approval Predictor
    </h1>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pickle.load(open("pipeline.pkl", "rb"))

model = load_model()

# ---------------- PERSONAL DETAILS ----------------
st.subheader("👤 Personal Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, value=25)
    income = st.number_input("Income", value=50000)
    emp_exp = st.number_input("Employment Experience (years)", value=2)

with col2:
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education", ['Master', 'High School', 'Bachelor', 'Associate', 'Doctorate'])

with col3:
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    loan_intent = st.selectbox("Loan Purpose", [
        'PERSONAL','EDUCATION','MEDICAL',
        'VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION'
    ])

# ---------------- LOAN DETAILS ----------------
st.subheader("💰 Loan Details")

col4, col5, col6 = st.columns(3)

with col4:
    loan_amount = st.number_input("Loan Amount", value=10000)
    interest_rate = st.number_input("Interest Rate (%)", value=10.0)
    loan_percent_income = st.number_input("Loan % of Income", value=0.2)

with col5:
    credit_history = st.number_input("Credit History Length", value=3)
    credit_score = st.number_input("Credit Score", value=650)

with col6:
    default = st.selectbox("Previous Default", ["Yes", "No"])

# ---------------- VALIDATION ----------------
def validate_inputs():
    if income <= 0:
        st.error("Income must be greater than 0")
        return False
    if loan_amount <= 0:
        st.error("Loan amount must be greater than 0")
        return False
    if credit_score <= 0:
        st.error("Credit score must be valid")
        return False
    return True

# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("Predict"):

    if not validate_inputs():
        st.stop()

    # ✅ CLEAN INPUT (MOST IMPORTANT PART)
    input_data = pd.DataFrame([{
        'person_age': float(age),
        'person_income': float(income),
        'person_emp_exp': float(emp_exp),
        'loan_amnt': float(loan_amount),
        'loan_int_rate': float(interest_rate),
        'loan_percent_income': float(loan_percent_income),
        'cb_person_cred_hist_length': float(credit_history),
        'credit_score': float(credit_score),
        'person_gender': str(gender),
        'person_education': str(education),
        'person_home_ownership': str(home),
        'loan_intent': str(loan_intent),
        'previous_loan_defaults_on_file': str(default)
    }])

    # ---------------- PREDICT ----------------
    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    # ---------------- OUTPUT ----------------
    st.subheader(f"📈 Approval Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # ---------------- RISK LEVEL ----------------
    if prob > 0.75:
        st.success("🟢 Low Risk Applicant")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk Applicant")
    else:
        st.error("🔴 High Risk Applicant")

    # ---------------- FEATURE IMPORTANCE ----------------
    try:
        if hasattr(model, "named_steps"):
            final_model = model.named_steps["model"]
        else:
            final_model = model

        if hasattr(final_model, "feature_importances_"):
            st.subheader("📊 Feature Importance")

            features = final_model.feature_names_in_
            importance = final_model.feature_importances_

            fig, ax = plt.subplots()
            ax.barh(features, importance)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")

            st.pyplot(fig)

    except:
        st.warning("Feature importance not available")

    # ---------------- LOGGING ----------------
    try:
        input_data['prediction'] = prediction
        input_data['probability'] = prob

        if not os.path.exists("logs.csv"):
            input_data.to_csv("logs.csv", index=False)
        else:
            input_data.to_csv("logs.csv", mode='a', header=False, index=False)

    except:
        pass


footer = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #111827;
    text-align: center;
    padding: 10px;
}
</style>

<div class="footer">
    Copyright by Aditya Paithankar
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
