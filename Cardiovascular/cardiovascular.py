import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

#loading dataset to a pandas Dataframe
cardiovascular_dataset = pd.read_csv('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/datasets/cardiovascular.csv')

random_seed = 42
sample_size = 70000
cardio = cardiovascular_dataset.sample(n=sample_size, random_state=random_seed)
print(cardio.shape)

columns_drop = ['cardio','id','age']
x = cardio.drop(columns = columns_drop,axis = 1)
y = cardio['cardio']

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

x_train,x_test,y_train,y_test = train_test_split(standardized_data,y,test_size = 0.2, random_state = 2)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train,y_train)

x_train_prediction = classifier.predict(x_train)
accuracy_x_train = accuracy_score(x_train_prediction,y_train)
print(accuracy_x_train)

x_test_prediction = classifier.predict(x_test)
accuracy_x_test = accuracy_score(x_test_prediction,y_test)
print(accuracy_x_test)

input_data = (1,163,72.0,135,80,1,2,0,0,0)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for only one instance (Earlier we trained and tested our model for too many instances but now only for one we want)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#now we have to standardize input data
std_data = scaler.transform(input_data_reshaped)

#prediction
prediction = classifier.predict(std_data)
print(prediction[0])

if (prediction[0] == 0):
  print("Not Cardiovascular problem")
else:
  print("Cardiovascular problem")

filename = 'trained_model.sav'
modelname = 'scaler.sav'
pickle.dump(classifier,open(filename,'wb'))
pickle.dump(scaler,open(modelname,'wb'))

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

# New input data
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
