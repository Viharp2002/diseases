import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Cardiovascular/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Cardiovascular/scaler.sav', 'rb'))


def prediction_System(input_data):

    # Convert the input data to a numpy array
    new_input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for only one instance
    new_input_data_reshaped = new_input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data using the same scaler used during training
    std_new_data = scaler.transform(new_input_data_reshaped)

    # Make predictions
    new_prediction = loaded_model.predict(std_new_data)

    if new_prediction[0] == 0:
        return "Not Cardiovascular problem"
    else:
        return "Cardiovascular problem"


def display_information():
    col1, col2 = st.columns(2)  # Create two columns

    # Column 1
    with col1:
        gender = st.text_input("Gender")
        Height = st.text_input("Height")
        Weight = st.text_input("Weight")
        Systolic_Blood_Pressure = st.text_input("Systolic Blood Pressure")
        Diastolic_Blood_Pressure = st.text_input("Diastolic Blood Pressure")

    # Column 2
    with col2:
        cholesterol = st.text_input("Cholesterol value")
        gluc = st.text_input("Glucose value")
        smoke = st.text_input('Does Person smoke? (yes/no)')
        active = st.text_input('Is Person alive? (yes/no)')
        alco = st.text_input('Does Person consume alcohol? (yes/no)')

    # more calculations
    lowercase_gender = gender.lower()
    lowercase_smoke = smoke.lower()
    lowercase_active = active.lower()
    lowercase_alco = alco.lower()

    if lowercase_gender == 'male':
       gender_value = 1
    else:
       gender_value = 2

    if lowercase_smoke == 'yes':
        smoke_value = 1
    else:
        smoke_value = 0

    if lowercase_active == 'yes':
        active_value = 1
    else:
        active_value = 0    
    
    if lowercase_alco == 'yes':
        alco_value = 1
    else:
        alco_value = 0
        
   # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Cardiovascular Disease Test Result'):
        diagnosis = prediction_System([gender_value, Height, Weight, Systolic_Blood_Pressure,
                                       Diastolic_Blood_Pressure, cholesterol, gluc,
                                       smoke_value, active_value, alco_value])

    st.success(diagnosis)