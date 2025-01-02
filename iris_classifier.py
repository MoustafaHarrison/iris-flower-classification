import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model
