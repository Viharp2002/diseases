import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Cardiovascular/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Cardiovascular/scaler.sav', 'rb'))

new_input_data = (1, 163, 72.0, 135, 80, 1, 2, 0, 0, 0)

# Convert the input data to a numpy array
new_input_data_as_numpy_array = np.asarray(new_input_data)

# Reshape the array as we are predicting for only one instance
new_input_data_reshaped = new_input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data using the same scaler used during training
std_new_data = scaler.transform(new_input_data_reshaped)

# Make predictions
new_prediction = loaded_model.predict(std_new_data)
print(new_prediction[0])

if new_prediction[0] == 0:
    print("Not Cardiovascular problem")
else:
    print("Cardiovascular problem")