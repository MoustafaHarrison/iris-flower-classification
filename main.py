import os
from iris_classifier import train_model
from iris_predict import make_predictions
import joblib


def main():
    # Step 1: Train the model
    print("Training the model...")
    model = train_model()  # This will call your training function from iris_classifier.py

    # Step 2: Save the model
    model_filename = 'iris_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

    # Step 3: Make predictions (you can modify this to take input or work with a new dataset)
    print("\nMaking predictions with the trained model...")
    predictions = make_predictions(model_filename)  # This calls the prediction function from iris_predict.py

    print("\nPredictions completed.")
    print(predictions)


if __name__ == "__main__":
    main()
