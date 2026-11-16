"""Run with: python -m unittest discover -s tests -v"""

import json
import tempfile
import unittest
from pathlib import Path

from src.model_service import ModelSpec, load_registry, normalize_sentiment, predict_review


class ExamplePipeline:
    classes_ = [0, 1]

    def predict(self, reviews):
        assert len(reviews) == 1
        return [1]

    def predict_proba(self, reviews):
        return [[0.15, 0.85]]


class StringPipeline:
    def predict(self, reviews):
        return ["Neutral"]


class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec(
            id="example",
            name="Test",
            artifact_path=Path("models/example.joblib"),
            feature="TF-IDF",
            description="",
            label_map={"0": "Negative", "1": "Positive"},
        )

    def test_binary_predictions_and_probabilities(self):
        result = predict_review(ExamplePipeline(), self.spec, "Good food")
        self.assertEqual(result.label, "Positive")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertAlmostEqual(result.probabilities["Negative"], 0.15)

    def test_three_class_string_labels(self):
        result = predict_review(StringPipeline(), self.spec, "It was okay")
        self.assertEqual(result.label, "Neutral")
        self.assertIsNone(result.confidence)

    def test_numeric_labels_require_explicit_map(self):
        with self.assertRaisesRegex(ValueError, "Unknown model label"):
            normalize_sentiment(1, {})

    def test_blank_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            predict_review(ExamplePipeline(), self.spec, "   ")

    def test_registry_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            manifest = root / "registry.json"
            manifest.write_text(json.dumps({"models": [{"id": "bad", "artifact": "../escape.joblib"}]}))
            with self.assertRaisesRegex(ValueError, "inside models"):
                load_registry(manifest, root)

    def test_registry_allows_missing_future_model_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            manifest = root / "registry.json"
            manifest.write_text(json.dumps({"models": [{"id": "a", "name": "A", "artifact": "models/a.joblib", "label_map": {}}]}))
            specs = load_registry(manifest, root)
            self.assertEqual(len(specs), 1)
            self.assertFalse(specs[0].artifact_path.exists())


if __name__ == "__main__":
    unittest.main()
