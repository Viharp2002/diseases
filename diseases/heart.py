import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Heart/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Heart/scaler.sav', 'rb'))


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
        return "Not Heart problem"
    else:
        return "Heart problem"
    
def display_information():
     
    col1, col2 , col3= st.columns(3)  # Create two columns

    # Column 1
    with col1:
        highBP = st.text_input("High Blood Pressure (yes/no)")
        highChol = st.text_input("High Cholesterol (yes/no)")
        BMI = st.text_input("Person's BMI")
        smoker = st.text_input("Smoker (yes/no)")
    # Column 2
    with col2:
        walk = st.text_input('Difficulty in Walking (yes/no)')
        gender = st.text_input("Gender")
        age = st.text_input("Age")
        diabetes = st.selectbox("Diabetes", ['never', 'no', 'moderate', 'high'])
        alco = st.text_input('Heavy Alcohol Consumption (yes/no)')


    with col3:
        physicactivity = st.text_input("Physical Activities (yes/no)")
        fruits = st.text_input('Consumes Fruits (yes/no)')
        veggies = st.text_input('Consumes Vegetables (yes/no)')
        stroke = st.text_input("Stroke (yes/no)")

    # more calculations
    lowercase_highBP = highBP.lower()
    lowercase_highChol = highChol.lower()
    lowercase_smoker = smoker.lower()
    lowercase_stroke = stroke.lower()
    lowercase_diabetes = diabetes.lower()
    lowercase_physicactivity = physicactivity.lower()
    lowercase_fruits = fruits.lower()
    lowercase_veggies = veggies.lower()
    lowercase_alco = alco.lower()
    lowercase_walk = walk.lower()
    lowercase_gender = gender.lower()


    if lowercase_highBP == 'yes':
       highBP_value = 1
    else:
       highBP_value = 0

    if lowercase_highChol == 'yes':
       highChol_value = 1
    else:
       highChol_value = 0

    if lowercase_smoker == 'yes':
       smoker_value = 1
    else:
       smoker_value = 0

    if lowercase_stroke == 'yes':
       stroke_value = 1
    else:
       stroke_value = 0

    if lowercase_diabetes == 'never':
       diabetes_value = 0
    elif lowercase_diabetes == 'no':
       diabetes_value = 0
    elif lowercase_diabetes == 'moderate':
       diabetes_value = 1
    else:
       diabetes_value = 2

    if lowercase_physicactivity == 'yes':
       physicactivity_value = 1
    else:
       physicactivity_value = 0

    if lowercase_fruits == 'yes':
       fruits_value = 1
    else:
       fruits_value = 0

    if lowercase_veggies == 'yes':
       veggies_value = 1
    else:
       veggies_value = 0

    if lowercase_alco == 'yes':
       alco_value = 1
    else:
       alco_value = 0

    if lowercase_walk == 'yes':
       walk_value = 1
    else:
       walk_value = 0

    if lowercase_gender == 'male':
       gender_value = 1
    else:
       gender_value = 0
   
    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        diagnosis = prediction_System([BMI, gender_value, age, alco_value, diabetes_value, fruits_value, highBP_value, highChol_value,
                                       physicactivity_value, smoker_value, stroke_value, veggies_value, walk_value])

    st.success(diagnosis)