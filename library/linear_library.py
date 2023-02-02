import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate, n_iterations):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(n_iterations):
            y_pred = X @ self.weights + self.bias
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return X @ self.weights + self.bias


import numpy as np

class PolynomialRegression():
    def __init__(self, degree):
        self.degree = degree
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate, n_iterations):
        X = self._transform_features(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(n_iterations):
            y_pred = X @ self.weights + self.bias
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        X = self._transform_features(X)
        return X @ self.weights + self.bias

    def _transform_features(self, X):
        X_poly = X.copy()
        for i in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** i))
        return X_poly