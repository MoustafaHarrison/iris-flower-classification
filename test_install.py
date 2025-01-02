import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Test NumPy
array = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", array)

# Test pandas
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print("Pandas DataFrame:\n", df)

# Test scikit-learn
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print("Train-Test Split:\nX_train:", X_train, "\nX_test:", X_test)
