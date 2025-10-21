import unittest
from src import preprocess, model

class TestTrainingPipeline(unittest.TestCase):
    def test_data_loading(self):
        X_train, X_test, y_train, y_test = preprocess.load_data("data/heart.csv")
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertGreater(len(y_train), 0)

    def test_model_creation(self):
        params = {"solver": "liblinear", "C": 1.0, "random_state": 42}
        clf = model.build_model(params)
        self.assertIsNotNone(clf)

if __name__ == "__main__":
    unittest.main()
