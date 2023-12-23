import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Thyroid/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Thyroid/scaler.sav', 'rb'))

 
# Input data for prediction
input_data = (72,1,0,0,0,0,0,0,0,1)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize input data
std_data = scaler.transform(input_data_reshaped)

# Prediction
prediction = loaded_model.predict(std_data)

if (prediction[0]==0):
  print("You have less chances to be affetced by Thyroid")
else:
  print("You have affected chances")
print(prediction[0])