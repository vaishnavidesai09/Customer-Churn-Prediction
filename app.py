import streamlit as st
import joblib as jb
import numpy as np
scaler = jb.load('scaleer.pkl')
model = jb.load("model.pkl")
st.title('Churn Predictor App')
st.divider()
st.write('Please enter the details in order to get predictions!')
st.divider()
age = st.number_input('Enter age',min_value=10,max_value=100,value=30)


tenure = st.number_input('Enter tenure',min_value=0,max_value=130,value=10)

monthly_charges = st.number_input('Enter Monthly Charges',min_value=30,max_value=150)

gender = st.selectbox("Enter gender ",['Male','Female'])
st.divider()
predictbutton = st.button('Predict')
st.divider()
if predictbutton:
    gender_selected = 1 if gender=='Female' else 0
    X= [age,gender_selected,tenure,monthly_charges]
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    predicted = 'Yes' if prediction==1 else 'No'
    st.balloons()
    st.write(f'Prediction: {predicted}')
else:
    st.write("please enter values to use predict button")    
