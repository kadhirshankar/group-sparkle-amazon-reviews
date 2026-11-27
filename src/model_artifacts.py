"""Retrieve only the team's checksum-pinned public M4 model release assets.

We never fetch review CSVs or train at app startup. Joblib uses pickle, so
only the pinned assets created by the team's verified GitHub Actions run
are permitted for remote loading.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


RELEASE_URL = (
    "https://github.com/kadhirshankar/group-sparkle-amazon-reviews/"
    "releases/download/m4-seed49-verified-v1"
)
MAX_MODEL_BYTES = 20_000_000

SHA256 = {
    "multinomial_naive_bayes_bow_pipeline.joblib":
        "2e0df98822add52e0d74f53c0e1e3172de2b5dbe5ce395b27a6c277719f305a3",
    "logistic_regression_tfidf_pipeline.joblib":
        "85782818206e536afbf5e5acc71bef738c3c51a8992dd53dc393a2130ef0adcb",
}


def ensure_model_artifact(path: Path) -> bool:
    """Download a missing trusted model, checking SHA-256 before installing it.

    Returns True only when a file was downloaded. Existing local artifacts
    remain untouched, including an optional locally trained neural model.
    """
    path = Path(path)
    if path.is_file():
        return False
    expected = SHA256.get(path.name)
    if expected is None:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_URL}/{path.name}"
    request = Request(url, headers={"User-Agent": "group-sparkle-cse3521/1.0"})
    tmp_name = None
    try:
        digest = hashlib.sha256()
        total = 0
        with urlopen(request, timeout=40) as response:
            with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".model-", delete=False) as output:
                tmp_name = output.name
                while True:
                    chunk = response.read(128 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_MODEL_BYTES:
                        raise ValueError("The downloaded model exceeds the safety size limit.")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Model SHA-256 mismatch; refusing to load downloaded pickle.")
        os.replace(tmp_name, path)
        return True
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)
