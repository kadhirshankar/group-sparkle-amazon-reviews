"""Tests for split integrity and sentiment-aware text preprocessing."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.text_preprocessing import clean_text, load_and_split, make_vectorizer


class TextPreprocessingTests(unittest.TestCase):
    def test_duplicate_review_text_cannot_cross_split(self):
        rows = []
        for score in range(1, 6):
            for index in range(10):
                rows.append({"Score": score, "Text": f"score {score} review {index}"})
        rows.append({"Score": 5, "Text": "score 5 review 0"})

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            train, test = load_and_split(path)

        self.assertTrue(set(train["Text"]).isdisjoint(set(test["Text"])))
        self.assertEqual(len(train) + len(test), 50)

    def test_clean_text_expands_negation(self):
        self.assertIn("is not", clean_text("This isn’t good<br />at all"))

    def test_unknown_vectorizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bow.*tfidf"):
            make_vectorizer("embeddings")


if __name__ == "__main__":
    unittest.main()
