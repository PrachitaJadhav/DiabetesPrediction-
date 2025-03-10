import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = r"C:\Users\Prachita Jadhav\Downloads\archive\diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

# Select only two features
X = df[['age', 'blood_glucose_level']]
y = df['diabetes']

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as model.pkl")
