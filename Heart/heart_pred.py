import pickle
import numpy as np

loaded_model = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Heart/trained_model.sav','rb'))
# Load the scaler used during training
scaler = pickle.load(open('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/Heart/scaler.sav', 'rb'))

new_input_data = (1.0,1.0,24.0,1.0,0.0,2.0,0.0,0.0,1.0,0.0,1.0,0.0,9.0)

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
    print("Not Heart problem")
else:
    print("Heart problem")