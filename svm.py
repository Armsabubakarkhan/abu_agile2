"""Decision Tree Classifier example to predict class for given marks."""

# pylint: disable=import-error
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    """Train a Decision Tree model and make a prediction."""
    features = [[30], [40], [50], [60], [20], [10], [70], [39]]
    labels = [0, 1, 1, 1, 0, 0, 1, 0]

    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(features, labels)

    test_marks = [[39]]
    prediction = classifier.predict(test_marks)

    print(prediction)


if __name__ == "__main__":
    main()
