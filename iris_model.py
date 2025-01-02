import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Importing joblib for saving the model

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model using joblib
joblib.dump(model, 'iris_model.joblib')
print("Model saved to iris_model.joblib")

# Load the saved model
model = joblib.load('iris_model.joblib')

# Example input (random values, replace with actual data as needed)
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict with the loaded model
prediction = model.predict(X_new)
print(f"Predicted class: {prediction}")

# If you want to predict new data, you can replace `X_new` with any input.
