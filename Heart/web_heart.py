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
    

def main():
    
    
    # giving a title
    st.title('Heart Disease Prediction Web App')
    
    # getting the input data from the user
    
    highBP = st.text_input("Does person affect from high BP? (yes/no)")
    highChol = st.text_input("Does person affect from high Cholesterol? (yes/no)")
    BMI = st.text_input("Person's BMI")
    smoker = st.text_input("Does Person smoke? (yes/no)")
    stroke = st.text_input("Is Person suffering from Stroke? (yes/no)")
    diabetes = st.text_input("Does Person have diabetes? (never/moderate/high)")
    physicactivity = st.text_input("Does Person do physical activities? (yes/no)" )
    fruits = st.text_input('Does Person consume fruits? (yes/no)')
    veggies = st.text_input('Does Person consume vegetables? (yes/no)')
    alco = st.text_input('Does Person consume heavy alcohol? (yes/no)')
    walk = st.text_input('Does Person find difficulty in walking? (yes/no)')
    gender = st.text_input("Person's Gender")
    age = st.text_input("Person's Age")

    #more calculations
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
    
    if st.button('Diabetes Test Result'):
        diagnosis = prediction_System([BMI, gender_value, age, alco_value, diabetes_value , fruits_value, highBP_value,highChol_value,
                                      physicactivity_value, smoker_value, stroke_value, veggies_value, walk_value])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()