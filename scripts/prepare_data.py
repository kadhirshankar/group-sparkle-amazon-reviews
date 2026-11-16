"""Create the team's reproducible 50,000-review sample.

Run from the repository root:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Score", "Text"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/Reviews.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/group_sample_seed49.csv")
    )
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def sentiment_counts(sample: pd.DataFrame) -> pd.Series:
    labels = pd.cut(
        sample["Score"],
        bins=[0, 2, 3, 5],
        labels=["Negative", "Neutral", "Positive"],
    )
    return labels.value_counts().reindex(["Positive", "Neutral", "Negative"])


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    reviews = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(reviews.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if len(reviews) < args.sample_size:
        raise ValueError(
            f"Input has {len(reviews):,} rows; cannot sample {args.sample_size:,}"
        )

    # pandas uses a deterministic RNG for a fixed version, seed, and input row order.
    sample = reviews.sample(n=args.sample_size, random_state=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    print(f"Wrote {len(sample):,} reviews to {args.output}")
    print(f"Sampling seed: {args.seed}")
    print(sentiment_counts(sample).to_string())
    print(f"Duplicate review texts: {sample.duplicated(subset='Text').sum():,}")


if __name__ == "__main__":
    main()
