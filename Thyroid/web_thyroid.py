import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Thyroid/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Thyroid/scaler.sav', 'rb'))

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
      return "You have less chances to be affetced by Thyroid"
    else:
      return "You have affected chances"
    

def main():
    
    
    # giving a title
    st.title('Thyroid Prediction Web App')
    
    
    # getting the input data from the user
    
    
    age = st.text_input("Person's Age")
    gender = st.text_input("Person's Gender")
    thyroxine = st.text_input("Does Person on thyroxine medication? (yes/no)")
    sick = st.text_input("Does person have sick today? (yes/no)")
    pregnant = st.text_input("Is Person pregnant? (yes/no)")
    thyroid_surgery = st.text_input("Does Person go through Thyroid Surgery earlier? (yes/no)")
    tumor = st.text_input("Does Person have a tumor related to the thyroid? (yes/no)")
    hypopituitary = st.text_input("Does Person have hypopituitarism? (yes/no)")
    psych = st.text_input("Does Person suffer from psychological issues? (yes/no)")
    TSH_measured = st.text_input("Do TSH (Thyroid Stimulating Hormone) levels have been measured? (yes/no)")
   
    #more calculations
    lowercase_gender = gender.lower()
    lowercase_thyroxine = thyroxine.lower()
    lowercase_sick = sick.lower()
    lowercase_pregnant = pregnant.lower()
    lowercase_thyroid_surgery = thyroid_surgery.lower()
    lowercase_tumor = tumor.lower()
    lowercase_hypopituitary = hypopituitary.lower()
    lowercase_psych = psych.lower()
    lowercase_TSH_measured = TSH_measured.lower()

    if lowercase_gender == 'male':
       gender_value = 1
    else:
       gender_value = 0

    if lowercase_thyroxine == 'yes':
        thyroxine_value = 1
    else:
        thyroxine_value = 0

    if lowercase_sick == 'yes':
        sick_value = 1
    else:
        sick_value = 0    
    
    if lowercase_pregnant == 'yes':
        pregnant_value = 1
    else:
        pregnant_value = 0    
    
    if lowercase_thyroid_surgery == 'yes':
        thyroid_surgery_value = 1
    else:
        thyroid_surgery_value = 0    
    
    if lowercase_tumor == 'yes':
        tumor_value = 1
    else:
        tumor_value = 0    
    
    if lowercase_hypopituitary == 'yes':
        hypopituitary_value = 1
    else:
        hypopituitary_value = 0    
    
    if lowercase_psych == 'yes':
        psych_value = 1
    else:
        psych_value = 0    
    
    if lowercase_TSH_measured == 'yes':
        TSH_measured_value = 1
    else:
        TSH_measured_value = 0    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Thyroid Test Result'):
        diagnosis = prediction_System([age,gender_value,thyroxine_value,sick_value,pregnant_value,thyroid_surgery_value,tumor_value,hypopituitary_value,psych_value,TSH_measured_value])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()