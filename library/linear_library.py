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



X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

test = LinearRegression()
test.fit(X, y, 0.01, 1000)
t = test.predict(X)
print(t)