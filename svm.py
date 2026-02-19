"""
SVM classification example using scikit-learn.
This script trains a simple SVM model and predicts results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

# Sample dataset
X_MARKS = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
Y_LABELS = [0, 0, 0, 1, 1]

# Create SVM model
MODEL = svm.SVC(kernel="linear")

# Train model
MODEL.fit(X_MARKS, Y_LABELS)

# Predict
PREDICTION = MODEL.predict([[2, 2]])

print("Prediction:", PREDICTION)

# Plotting (so imports are used)
plt.scatter(X_MARKS[:, 0], X_MARKS[:, 1], c=Y_LABELS)
plt.title("SVM Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
