"""Train Logistic Regression with TF-IDF features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_preprocessing import load_and_split, make_vectorizer  # noqa: E402
from src.training_utils import train_and_evaluate  # noqa: E402

MODEL_NAME = "logistic_regression_tfidf"
MODEL_FILENAME = "logistic_regression_tfidf_pipeline.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", make_vectorizer("tfidf")),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced"),
            ),
        ]
    )


def run(train, test, output_dir: Path, models_dir: Path):
    return train_and_evaluate(
        MODEL_NAME,
        build_pipeline(),
        train,
        test,
        output_dir,
        models_dir / MODEL_FILENAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/group_sample_seed49.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m2"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train, test = load_and_split(args.data)
    metrics, _ = run(train, test, args.output_dir, args.models_dir)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
