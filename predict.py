from __future__ import print_function
import pandas as pd
import numpy as np
import joblib

# Load the model from the file
RF = joblib.load('random_forest_model.pkl')

# Get user input for prediction
N = float(input("Enter the value for N: "))
P = float(input("Enter the value for P: "))
K = float(input("Enter the value for K: "))
temperature = float(input("Enter the value for temperature: "))
humidity = float(input("Enter the value for humidity: "))
ph = float(input("Enter the value for ph: "))
rainfall = float(input("Enter the value for rainfall: "))

data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
prediction = RF.predict(data)
print("The prediction is:", prediction)
