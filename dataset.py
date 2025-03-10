import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Specify the file path
file_path = r"C:\Users\Prachita Jadhav\Downloads\archive\diabetes_prediction_dataset.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())
print("Dataset Preview:\n", df.head())

# Step 1: Define Features (X) and Target (y)
X = df.drop(columns=['diabetes'])  # All columns except 'diabetes'
y = df['diabetes']  # Target variable

# Step 2: Convert Categorical Data to Numeric
encoder = LabelEncoder()

# Encoding 'gender' and 'smoking_history'
X['gender'] = encoder.fit_transform(X['gender'])  
X['smoking_history'] = encoder.fit_transform(X['smoking_history'])

# Step 3: Split Data into Training and Testing Sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of the datasets
print(f"Training Data: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing Data: X_test: {X_test.shape}, y_test: {y_test.shape}")
# Step 4: Define a Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Now, predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
