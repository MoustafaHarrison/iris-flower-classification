import numpy as np
import joblib

# Load the saved model
model = joblib.load('iris_model.joblib')


# Function to predict the class of a new input
def predict_iris_class(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare the input data
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict the class
    prediction = model.predict(X_new)

    # Return the predicted class
    return prediction[0]


# CLI to get input from the user
if __name__ == "__main__":
    print("Iris Flower Prediction")

    # Take user inputs
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    # Make prediction
    predicted_class = predict_iris_class(sepal_length, sepal_width, petal_length, petal_width)

    # Display the prediction
    print(f"The predicted class for the given Iris flower is: {predicted_class}")
