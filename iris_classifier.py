import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline

# Set up logging for tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Step 1: Load the Iris dataset
def load_data():
    """Load the Iris dataset"""
    try:
        data = load_iris()
        X = data.data  # Features: sepal length, sepal width, petal length, petal width
        y = data.target  # Target: Iris species
        logger.info("Iris dataset loaded successfully.")
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# Step 2: Split the data into training and testing sets
def split_data(X, y, test_size=0.3, random_state=42):
    """Split the dataset into training and testing sets"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(
            f"Data split into training and testing sets: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


# Step 3: Create a pipeline with scaling and RandomForest classifier
def create_model():
    """Create a pipeline that includes scaling and RandomForestClassifier"""
    try:
        model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
        logger.info("Model pipeline created with scaling and RandomForest.")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


# Step 4: Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(model, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV"""
    try:
        param_grid = {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [None, 10, 20, 30],
            'randomforestclassifier__min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error tuning hyperparameters: {e}")
        raise


# Step 5: Train the model
def train_model(X_train, y_train):
    """Train the model on the training data"""
    try:
        model = create_model()
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training the model: {e}")
        raise


# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Model accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{report}")

        return accuracy, cm, report
    except Exception as e:
        logger.error(f"Error evaluating the model: {e}")
        raise


# Step 7: Save the model to a file
def save_model(model, filename='iris_model.joblib'):
    """Save the trained model to a file"""
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise


# Step 8: Predict with the trained model (optional)
def predict(model, X_test):
    """Make predictions with the trained model"""
    try:
        predictions = model.predict(X_test)
        logger.info(f"Predictions completed: {predictions[:5]}")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


# Main workflow
def main():
    try:
        # Load data
        X, y = load_data()

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train the model
        logger.info("Training the model...")
        model = train_model(X_train, y_train)

        # Tune hyperparameters
        logger.info("Tuning hyperparameters...")
        best_model = tune_hyperparameters(model, X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating the model...")
        evaluate_model(best_model, X_test, y_test)

        # Save the model
        save_model(best_model)

        # Optionally, make predictions
        predictions = predict(best_model, X_test)

    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")


if __name__ == '__main__':
    main()
