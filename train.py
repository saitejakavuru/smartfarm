from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the CSV data
df = pd.read_csv('Crop_recommendation.csv')

# Define features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
print("RF's Accuracy is: ", x)

# Save the model to a file
joblib.dump(RF, 'random_forest_model.pkl')
print("Model saved to random_forest_model.pkl")
