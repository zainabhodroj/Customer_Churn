import streamlit as st
import pandas as pd


def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior = st.sidebar.selectbox('Senior Citizen', (1, 0))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.sidebar.slider('Tenure', 0, 72, 47)
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multi_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
    internet = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_sec = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_back = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    dev_protect = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_sup = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    stream_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    stream_movie = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_bill = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    payment = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check',
                                                      'Bank transfer (automatic)',
                                                      'Credit card (automatic)'))
    monthly_charge = st.sidebar.number_input('Monthly Charges', 18.25, 119.0, 56.75)
    total_charge = st.sidebar.number_input('Total Charges', 0.0, 8684.8, 1556.75)
    # Creating DataFrame
    data = {'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multi_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_back,
            'DeviceProtection': dev_protect,
            'TechSupport': tech_sup,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_movie,
            'Contract': contract,
            'PaperlessBilling': paperless_bill,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charge,
            'TotalCharges': total_charge}
    features = pd.DataFrame(data, index=[0])
    return features
