"""Tests for DistilBERT batching (fine-tuning itself is too slow for a unit test)."""

import unittest

import numpy as np

from scripts.train_distilbert import MAX_TOKENS, batches


class DistilBertBatchTests(unittest.TestCase):
    def test_batches_have_matching_tokens_labels_and_padding_mask(self):
        texts = ["great"] * 19 + ["this coffee was stale, tasteless, and arrived late"]
        labels = np.array([2] * 19 + [0])
        all_batches = list(batches(texts, labels))

        self.assertEqual([len(y) for _, y in all_batches], [16, 4])
        tokens, y = all_batches[-1]
        self.assertEqual(tokens["input_ids"].shape, tokens["attention_mask"].shape)
        self.assertEqual(y.tolist(), [2, 2, 2, 0])
        self.assertEqual(tokens["attention_mask"][0, -1].item(), 0)  # short review is padded
        self.assertEqual(tokens["attention_mask"][-1].min().item(), 1)  # longest review has no padding

    def test_long_reviews_are_cut_to_max_tokens(self):
        tokens, _ = next(batches(["delicious " * 1000], np.array([2])))
        self.assertEqual(tokens["input_ids"].shape[1], MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
