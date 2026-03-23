# np.dot(a, b) multiplies matrices. It is similar to a @ b
# X.T (X transpose)

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


class CustomLogisticRegression:
    '''
    A simple Logistic Regression implementation using GradientDescent.
    '''
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def _sigmoid(self, z):
        # Sigmoid function maps linear output to probability (0 to 1)
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        # Initialize parameters (weights and bias)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            # Linear model: z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        # Threshold at 0.5
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


def run_comparison():
    # Load Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale features
    # StandardScaler scales transforms to have a mean of 0 and standard dev of 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train custom model
    custom_model = CustomLogisticRegression(learning_rate=0.1, epochs=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)

    custom_accuracy = np.mean(custom_preds == y_test)

    # Train Scikit-Learn Model
    sklearn_model = SklearnLogisticRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_accuracy = np.mean(sklearn_preds == y_test)

    # Results
    print(f"Custom model accuracy: {custom_accuracy:.2f}")
    print(f"Sklearn model accuracy: {sklearn_accuracy:.2f}")


if __name__ == "__main__":
    run_comparison()
