import joblib
import numpy as np


def make_predictions(model_filename):
    # Load the trained model from the joblib file
    model = joblib.load(model_filename)

    # Example test data (You can replace this with new input data)
    test_data = np.array([[5.1, 3.5, 1.4, 0.2],  # Example Iris Setosa sample
                          [6.7, 3.1, 4.4, 1.4]])  # Example Iris Versicolor sample

    # Make predictions using the trained model
    predictions = model.predict(test_data)

    return predictions
