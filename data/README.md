# Data

We use [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

The project sample is 50,000 reviews with seed 49. The original dataset, sample, and generated CSVs are excluded from GitHub by `.gitignore`.

From the repository root, download/extract the Kaggle dataset so the original file is `data/Reviews.csv`, then run:

```sh
python scripts/prepare_data.py
```

This creates `data/group_sample_seed49.csv`. Model training removes duplicate review text before the stratified 80/20 split so identical text cannot leak across the two sets.
