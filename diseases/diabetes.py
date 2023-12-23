# disease1.py
import streamlit as st
import pickle
import numpy as np


loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Diabetes/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Diabetes/scaler.sav', 'rb'))

def prediction_System(input_data):

    # Convert the input data to a numpy array
    new_input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for only one instance
    new_input_data_reshaped = new_input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the same scaler used during training
    std_new_data = scaler.transform(new_input_data_reshaped)

    # Make predictions
    prediction = loaded_model.predict(std_new_data)

    if (prediction[0]==0):
      return "You have less chances to be affetced by Diabetes"
    else:
      return "You have affected chances"
    

def display_information():
    st.write("Information about Disease 1 goes here.")
    
     # getting the input data from the user
    
    
    age = st.text_input("Person's Age")
    hypertension = st.text_input("Person's Hypertension value")
    heart = st.text_input("Does person affect from Heart Disease? (yes/no)")
    smoke = st.text_input("Does person have smoking history? (yes/no)")
    bmi = st.text_input("Person's BMI value")
    HbA1c_level = st.text_input("Person's HbA1c level")
    blood_glucose_level = st.text_input("Person's Blood Glucose level" )
    gender = st.text_input("Person's Gender")

    #more calculations
    lowercase_gender = gender.lower()
    lowercase_smoke = smoke.lower()
    lowercase_heart = heart.lower()

    if lowercase_gender == 'male':
       gender_value = 1
    else:
       gender_value = 0

    if lowercase_smoke == 'yes':
        smoke_value = 1
    else:
        smoke_value = 0

    if lowercase_heart == 'yes':
        heart_value = 1
    else:
        heart_value = 0    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = prediction_System([gender_value,heart_value,age,hypertension,smoke_value,bmi,HbA1c_level,blood_glucose_level])
        
        
    st.success(diagnosis)
