import unittest
import numpy as np
from library.linear_library import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegression()
        self.X = np.array([[1], [2], [3], [4]])
        self.y = np.array([2, 4, 6, 8])
        self.learning_rate = 0.01
        self.n_iterations = 10000

    def test_fit(self):
        self.model.fit(self.X, self.y, self.learning_rate, self.n_iterations)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)

    def test_predict(self):
        self.model.fit(self.X, self.y, self.learning_rate, self.n_iterations)
        y_pred = self.model.predict(self.X)
        self.assertTrue(np.allclose(y_pred, self.y, 0.1))

if __name__ == '__main__':
    unittest.main()
