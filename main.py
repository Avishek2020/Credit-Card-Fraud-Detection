import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('./fraud_detection_model.pkl')

# Set up the Streamlit app
st.title('Fraud Detection App')

# Input fields for all 29 features
st.write("Enter the transaction details to check for fraud:")
V1 = st.number_input('V1')
V2 = st.number_input('V2')
V3 = st.number_input('V3')
V4 = st.number_input('V4')
V5 = st.number_input('V5')
V6 = st.number_input('V6')
V7 = st.number_input('V7')
V8 = st.number_input('V8')
V9 = st.number_input('V9')
V10 = st.number_input('V10')
V11 = st.number_input('V11')
V12 = st.number_input('V12')
V13 = st.number_input('V13')
V14 = st.number_input('V14')
V15 = st.number_input('V15')
V16 = st.number_input('V16')
V17 = st.number_input('V17')
V18 = st.number_input('V18')
V19 = st.number_input('V19')
V20 = st.number_input('V20')
V21 = st.number_input('V21')
V22 = st.number_input('V22')
V23 = st.number_input('V23')
V24 = st.number_input('V24')
V25 = st.number_input('V25')
V26 = st.number_input('V26')
V27 = st.number_input('V27')
V28 = st.number_input('V28')
Amount_scaled = st.number_input('Amount_scaled')

# Prediction button
if st.button('Predict Fraud'):
    # Create feature array for prediction
    features = np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, 
                         V11, V12, V13, V14, V15, V16, V17, V18, 
                         V19, V20, V21, V22, V23, V24, V25, V26, 
                         V27, V28, Amount_scaled]).reshape(1, -1)
    
    # Predict using the loaded model
    prediction = model.predict(features)
    
    # Display result
    if prediction[0] == 1:
        st.error('This transaction is likely to be fraudulent!')
    else:
        st.success('This transaction appears to be legitimate.')
