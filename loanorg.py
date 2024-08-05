import joblib
import pandas as pd
import numpy as np
import streamlit as st
import xgboost
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load saved preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
xgb = joblib.load('best_model.pkl')

# Function to preprocess input data
def data_preprocess(age, income, loan_amount, credit_score, months_employed, interest_rate,
                    dti_ratio, education, employment_type, marital_status, has_mortgage, loan_purpose,
                    has_cosigner, has_dependents):
    data = pd.DataFrame({
        'Age': [age], 'Income': [income], 'LoanAmount': [loan_amount],
        'CreditScore': [credit_score], 'MonthsEmployed': [months_employed],
        'InterestRate': [interest_rate], 'DTIRatio': [dti_ratio], 'Education': [education],
        'EmploymentType': [employment_type], 'MaritalStatus': [marital_status],
        'HasMortgage': [has_mortgage], 'LoanPurpose': [loan_purpose],
        'HasCoSigner': [has_cosigner], 'HasDependents': [has_dependents]
    })

    # Apply the same preprocessing pipeline used during training
    data_preprocessed = preprocessor.transform(data)
    data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=preprocessor.get_feature_names_out())
    
    # Strip prefixes from column names
    data_preprocessed_df.columns = [col.split('__')[-1] for col in data_preprocessed_df.columns]
    
    return data_preprocessed_df

def main():
    st.image("loan pic.jpg")
    st.title('This is a Loan Default Prediction App')

    st.markdown("---")
    st.markdown("**This App will predict whether the customer will default on loan repayment or not**.")
   
    st.sidebar.markdown("##### *Enter customer information below* :")
    st.sidebar.divider()

    age = st.sidebar.slider('Age', min_value=18, max_value=100, value=25, step=1)
    income = st.sidebar.number_input('Income', value=50000, step=1000)
    loan_amount = st.sidebar.number_input('Loan Amount', value=10000, step=100)
    credit_score = st.sidebar.number_input('Credit Score', value=600, step=10)
    months_employed = st.sidebar.number_input('Months Employed', value=12, step=1)
    interest_rate = st.sidebar.slider('Interest Rate(%)', 0.0, 25.0, 2.0)
    dti_ratio = st.sidebar.slider('DTI Ratio', 0.0, 1.0, 0.3)
    education = st.sidebar.selectbox("Education", ["Bachelor's", "Master's", 'High School', 'PhD'])
    employment_type = st.sidebar.selectbox('Employment Type', ['Full-time', 'Unemployed', 'Self-employed', 'Part-time'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
    has_mortgage = st.sidebar.radio("Has Mortgage", ['No', 'Yes'])
    loan_purpose = st.sidebar.selectbox('Loan Purpose', ['Other', 'Auto', 'Business', 'Home', 'Education'])
    has_cosigner = st.sidebar.radio("Has Cosigner", ['No', 'Yes'])
    has_dependents = st.sidebar.radio("Has Dependents", ['No', 'Yes'])

    st.markdown("---")
    st.markdown("##### *View your answer*")

    # Initialize prediction and prediction_proba
    prediction = None
    prediction_proba = None

    if st.sidebar.button('Predict'):
        user_data = data_preprocess(age, income, loan_amount, credit_score, months_employed, interest_rate,
                                    dti_ratio, education, employment_type, marital_status, has_mortgage,
                                    loan_purpose, has_cosigner, has_dependents)
        
        prediction = xgb.predict(user_data)
        prediction_proba = xgb.predict_proba(user_data)
        
        # Display the results
        st.subheader(f'Prediction: {prediction[0]}') 
        st.write("1 means Loan will default --//-- 0 means Loan will not default") 
        st.subheader('Prediction Probability')
        st.write(prediction_proba)

        if prediction[0] == 1:
            st.error("Risk of default: High")
        else:
            st.success("Risk of default: Low")

if __name__ == "__main__":
    main()
