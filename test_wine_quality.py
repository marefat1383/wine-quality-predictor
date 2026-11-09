import unittest

from wine_quality import predict_quality


class WineQualityTests(unittest.TestCase):
    def test_prediction_has_valid_labels(self):
        predicted, actual = predict_quality(0)
        self.assertIn(predicted, {"standard", "high"})
        self.assertIn(actual, {"standard", "high"})

    def test_invalid_index_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_quality(-1)


if __name__ == "__main__":
    unittest.main()
