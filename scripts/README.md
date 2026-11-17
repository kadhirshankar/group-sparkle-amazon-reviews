# Scripts

Run these commands from the repository root:

```sh
python scripts/prepare_data.py
python scripts/train_m2_baselines.py
```

The first command creates the shared 50,000-row seed-49 sample. The second runs both model files, assembles their comparison, and writes M2 tables, figures, and error-analysis examples under `outputs/m2/`.

Each model can also be trained independently:

```sh
python scripts/train_naive_bayes.py
python scripts/train_logistic_regression.py
```

The generated model pipelines and report artifacts are intentionally excluded from Git.
