"""Tests for the neural network: training runs, the NumPy export matches PyTorch, and the app can use it."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.train_neural_network import export_pipeline, train_model
from src.model_service import ModelSpec, predict_review

TEXTS = ["great tasty snack", "awful stale mess", "it was okay", "love this coffee", "terrible taste", "fine nothing special"] * 5
LABELS = np.array([2, 0, 1, 2, 0, 1] * 5)  # 0 = Negative, 1 = Neutral, 2 = Positive


class NeuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TfidfVectorizer().fit(TEXTS)
        X = cls.vectorizer.transform(TEXTS)
        cls.model, cls.history = train_model((X, LABELS), (X, LABELS), hidden1=8, hidden2=4, max_epochs=3)
        cls.pipeline = export_pipeline(cls.model, cls.vectorizer, X, TEXTS)  # raises if NumPy != PyTorch

    def test_training_records_one_history_row_per_epoch(self):
        self.assertEqual([row["epoch"] for row in self.history], list(range(1, len(self.history) + 1)))

    def test_probabilities_are_valid(self):
        probs = self.pipeline.predict_proba(["great snack", "awful"])
        self.assertEqual(probs.shape, (2, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)

    def test_app_can_use_the_pipeline(self):
        spec = ModelSpec("nn", "Feedforward NN", Path("models/nn.joblib"), "TF-IDF", "", {})
        prediction = predict_review(self.pipeline, spec, "great tasty coffee")
        self.assertIn(prediction.label, {"Negative", "Neutral", "Positive"})
        self.assertEqual(set(prediction.probabilities), {"Negative", "Neutral", "Positive"})

    def test_saved_file_gives_same_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nn.joblib"
            joblib.dump(self.pipeline, path)
            loaded = joblib.load(path)
        np.testing.assert_allclose(loaded.predict_proba(TEXTS), self.pipeline.predict_proba(TEXTS))


if __name__ == "__main__":
    unittest.main()
