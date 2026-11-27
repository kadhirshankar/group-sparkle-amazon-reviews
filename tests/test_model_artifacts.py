"""Offline tests for the signed-off M4 model download contract."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import model_artifacts


class ReleaseDownloadTests(unittest.TestCase):
    def test_known_release_models_are_pinned(self):
        self.assertEqual(len(model_artifacts.SHA256), 2)
        for name, checksum in model_artifacts.SHA256.items():
            self.assertTrue(name.endswith(".joblib"))
            self.assertEqual(len(checksum), 64)

    def test_existing_local_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multinomial_naive_bayes_bow_pipeline.joblib"
            path.write_bytes(b"local trusted model")
            with patch.object(model_artifacts, "urlopen") as request:
                self.assertFalse(model_artifacts.ensure_model_artifact(path))
                request.assert_not_called()
            self.assertEqual(path.read_bytes(), b"local trusted model")

    def test_unregistered_model_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(model_artifacts, "urlopen") as request:
                self.assertFalse(model_artifacts.ensure_model_artifact(Path(folder) / "unknown.joblib"))
                request.assert_not_called()

    def test_verifies_sha_before_writing(self):
        payload = b"test model contents"
        filename = "multinomial_naive_bayes_bow_pipeline.joblib"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / filename
            with patch.dict(model_artifacts.SHA256, {filename: hashlib.sha256(payload).hexdigest()}):
                with patch.object(model_artifacts, "urlopen", return_value=io.BytesIO(payload)) as request:
                    self.assertTrue(model_artifacts.ensure_model_artifact(path))
                    self.assertEqual(path.read_bytes(), payload)
                    self.assertIn(filename, request.call_args.args[0].full_url)

    def test_rejects_wrong_hash_and_leaves_no_artifact(self):
        filename = "logistic_regression_tfidf_pipeline.joblib"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / filename
            with patch.object(model_artifacts, "urlopen", return_value=io.BytesIO(b"not a model")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    model_artifacts.ensure_model_artifact(path)
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(folder).glob(".model-*")))


if __name__ == "__main__":
    unittest.main()
