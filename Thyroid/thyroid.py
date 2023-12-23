import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
thyroid_data = pd.read_csv('C:/Users/vihar/OneDrive/Desktop/stuff/extras/Models - MedVistaHub/datasets/thyroid.csv')

random_seed = 42
sample_size = 3770

diabetes_dataset = thyroid_data.sample(n=sample_size, random_state=random_seed)

# Separate features and target variable
column_drop = ["query on thyroxine","on antithyroid medication","I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","TSH","T3 measured","T3","TT4 measured","TT4","T4U measured","T4U","FTI measured","FTI","TBG measured","TBG","referral source","binaryClass"]
x = diabetes_dataset.drop(columns=column_drop, axis=1)
y = diabetes_dataset['binaryClass']  

# Standardize the features
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Use a RandomForestClassifier instead of SVM
classifier = svm.SVC(kernel='linear', class_weight='balanced')
classifier.fit(x_train, y_train)


# Accuracy on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print(training_data_accuracy)

# Accuracy on test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(test_data_accuracy)




# Input data for prediction
input_data = (72,1,0,0,0,0,0,0,0,1)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize input data
std_data = scaler.transform(input_data_reshaped)

# Prediction
prediction = classifier.predict(std_data)

if (prediction[0]==0):
  print("You have less chances to be affetced by Thyroid")
else :
  print("You have chances to be affected by Thyroid")

print(prediction[0])

filename = 'trained_model.sav'
modelname = 'scaler.sav'
pickle.dump(classifier,open(filename,'wb'))
pickle.dump(scaler,open(modelname,'wb'))

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

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
  print("You have chances to be affected by Thyroid")
print(prediction[0])