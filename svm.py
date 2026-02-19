"""
SVM classification example using scikit-learn.

This script trains a simple Support Vector Machine model
and visualizes the classification data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def main():
    """Train SVM model and display prediction + plot."""

    # Sample dataset
    x_marks = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y_labels = [0, 0, 0, 1, 1]

    # Create model
    model = svm.SVC(kernel="linear")

    # Train
    model.fit(x_marks, y_labels)

    # Predict
    prediction = model.predict([[2, 2]])
    print("Prediction:", prediction)

    # Plot (so matplotlib import is used)
    plt.scatter(x_marks[:, 0], x_marks[:, 1], c=y_labels)
    plt.title("SVM Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == "__main__":
    main()
