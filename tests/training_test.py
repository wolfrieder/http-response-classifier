import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


# Created by ChatGPT4
class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Simplifying the problem to a binary classification (class 0 vs others)
        y = (y == 0).astype(int)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=10
        )

        # Create and train the logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def test_model_accuracy(self):
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Check accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, 0.8, "Accuracy should be at least 80%")

    def test_classification_report(self):
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Print classification report
        report = classification_report(self.y_test, y_pred)
        print("\nClassification Report:\n", report)

    def test_confusion_matrix_plot(self):
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Class 0", "Not Class 0"],
            yticklabels=["Class 0", "Not Class 0"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
