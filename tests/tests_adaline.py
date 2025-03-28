import unittest

import numpy as np

from adaline.adaline_np import *


class TestAdaLine(unittest.TestCase):
    def test_activation_function(self):
        x = np.array([1, -1, 0.5])
        np.testing.assert_array_equal(activation_function(x), x)

    def test_initialize_weights(self):
        weights = initialize_weights(3)
        self.assertEqual(weights.shape, (4,))

    def test_predict(self):
        X = np.array([[1, 2], [3, 4]])
        weights = np.array([0.1, 0.2, 0.3])
        preds = predict(X, weights)
        expected = np.array([0.1 + 1 * 0.2 + 2 * 0.3, 0.1 + 3 * 0.2 + 4 * 0.3])
        np.testing.assert_allclose(preds, expected)

    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.0])
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.00666666, places=5)

    def test_update_weights(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        weights = np.array([0.1, 0.2, 0.3])
        new_weights = update_weights(X, y, weights, 0.01)
        self.assertEqual(new_weights.shape, weights.shape)

    def test_train(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        weights = train(X, y, epochs=10, lr=0.01)
        self.assertEqual(weights.shape, (X.shape[1] + 1,))


if __name__ == "__main__":
    unittest.main()
