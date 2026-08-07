import streamlit as st
import numpy as np

st.title("Credit Risk Default Prediction Dashboard")
st.write("Enter applicant details to predict loan default probability.")

# User Inputs
grade = st.selectbox("Select Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
dti = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=100.0, value=15.0)
annual_inc = st.number_input("Annual Income ($)", min_value=1000.0, value=60000.0)
loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, value=15000.0)
emp_years = st.selectbox("Employment Length", ['< 1 year', '1-5 years', '8+ years'])

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
g_num = grade_map[grade]

# Simple scoring logic based on trained odds ratios
if st.button("Calculate Risk Score"):
    risk_score = (g_num * 0.44) + (0.1 if dti > 25 else 0) - (0.09 if emp_years == '8+ years' else 0)
    probability = 1 / (1 + np.exp(-(-2.0 + risk_score)))
    
    st.subheader(f"Predicted Default Probability: {probability:.2%}")
    if probability > 0.30:
        st.error("High Risk: Loan application likely to default. Recommend Rejection or Higher Interest Rate.")
    else:
        st.success("Low Risk: Applicant profile is stable. Recommend Approval.")