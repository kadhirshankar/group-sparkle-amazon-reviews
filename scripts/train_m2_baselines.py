"""Run both classical M2 models and assemble their comparison artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_logistic_regression import run as run_logistic_regression  # noqa: E402
from scripts.train_naive_bayes import run as run_naive_bayes  # noqa: E402
from src.text_preprocessing import load_and_split, make_vectorizer  # noqa: E402
from src.training_utils import LABELS, train_and_evaluate  # noqa: E402


def save_informative_features(train, representation: str, output_dir: Path) -> None:
    """Use Logistic Regression coefficients to explain one representation."""
    pipeline = Pipeline(
        [
            ("vectorizer", make_vectorizer(representation)),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced"),
            ),
        ]
    )
    pipeline.fit(train["Text"], train["sentiment"])
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    rows = []
    for label in ("Positive", "Negative"):
        coefficients = classifier.coef_[list(classifier.classes_).index(label)]
        indices = np.argsort(coefficients)[-10:][::-1]
        for rank, index in enumerate(indices, start=1):
            rows.append(
                {
                    "representation": representation,
                    "class": label,
                    "rank": rank,
                    "feature": feature_names[index],
                    "coefficient": coefficients[index],
                }
            )

    features = pd.DataFrame(rows)
    features.to_csv(
        output_dir / f"{representation}_informative_features.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, label, color in zip(
        axes, ("Positive", "Negative"), ("#277da1", "#d1495b")
    ):
        subset = features[features["class"] == label].sort_values("coefficient")
        ax.barh(subset["feature"], subset["coefficient"], color=color)
        ax.set_title(f"Features predicting {label.lower()} sentiment")
        ax.set_xlabel("Logistic regression coefficient")
    fig.suptitle(f"Top informative {representation.upper()} features")
    fig.tight_layout()
    fig.savefig(output_dir / f"{representation}_informative_features.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/group_sample_seed49.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m2"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--error-count", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_and_split(args.data)
    metadata = {
        "source_sample_rows": len(pd.read_csv(args.data, usecols=["Text"])),
        "rows_after_duplicate_removal": len(train) + len(test),
        "training_rows": len(train),
        "test_rows": len(test),
        "split_seed": 49,
        "test_fraction": 0.2,
        "labels": LABELS,
        "training_class_counts": train["sentiment"].value_counts().to_dict(),
        "test_class_counts": test["sentiment"].value_counts().to_dict(),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    naive_bayes_metrics, _ = run_naive_bayes(
        train, test, args.output_dir, args.models_dir
    )
    logistic_metrics, logistic_predictions = run_logistic_regression(
        train, test, args.output_dir, args.models_dir
    )

    for representation in ("bow", "tfidf"):
        save_informative_features(train, representation, args.output_dir)

    majority_pipeline = Pipeline(
        [
            ("vectorizer", make_vectorizer("bow")),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    majority_metrics, _ = train_and_evaluate(
        "majority_class_baseline", majority_pipeline, train, test, args.output_dir
    )

    metrics = pd.DataFrame(
        [naive_bayes_metrics, logistic_metrics, majority_metrics]
    ).sort_values("macro_f1", ascending=False)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)

    error_mask = test["sentiment"].to_numpy() != logistic_predictions
    columns = [
        column
        for column in ("Id", "Score", "Summary", "Text", "sentiment")
        if column in test
    ]
    errors = test.loc[error_mask, columns].copy()
    errors["predicted_sentiment"] = logistic_predictions[error_mask]
    errors["review_word_count"] = errors["Text"].fillna("").str.split().str.len()
    errors.sample(n=min(args.error_count, len(errors)), random_state=49).to_csv(
        args.output_dir / "misclassified_examples.csv", index=False
    )

    print(f"Training rows: {len(train):,}; test rows: {len(test):,}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
