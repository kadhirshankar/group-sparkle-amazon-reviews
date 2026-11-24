"""Lightweight data-contract tests for reported M3 dashboard metrics."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M3ReportMetricsTests(unittest.TestCase):
    def test_dashboard_metrics_have_required_fields(self):
        data = json.loads((ROOT / "report" / "model_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(data["test_rows"], 9279)
        self.assertEqual(data["split_seed"], 49)
        self.assertEqual(len(data["models"]), 5)
        for entry in data["models"]:
            for name in ("model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "training_seconds"):
                self.assertIn(name, entry)
            for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                self.assertGreaterEqual(entry[name], 0)
                self.assertLessEqual(entry[name], 1)
            self.assertGreater(entry["training_seconds"], 0)

    def test_distilbert_is_best_macro_f1(self):
        models = json.loads((ROOT / "report" / "model_metrics.json").read_text(encoding="utf-8"))["models"]
        best = max(models, key=lambda item: item["macro_f1"])
        self.assertEqual(best["model"], "DistilBERT (Fine-Tuned)")


if __name__ == "__main__":
    unittest.main()
