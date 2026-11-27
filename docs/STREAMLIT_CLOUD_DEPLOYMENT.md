# Deploy Group Sparkle to Streamlit Community Cloud

## Status
Public Streamlit app launched: https://group-sparkle-amazon-reviews-cpxkdw3k7qspxcssk2glkh.streamlit.app/
The November 26 deployment screenshot shows **2 / 3** registered models ready: Multinomial Naive Bayes (BoW) and Logistic Regression (TF-IDF). The optional feedforward neural network is not currently hosted. A live public prediction click still needs a final browser check.

The classical-model release exists and its checksums are pinned in `src/model_artifacts.py`.
The app can download and verify its **two real trained model pipelines** on a
fresh checkout. Neither the raw Amazon CSV nor full training libraries are
needed on the deployed server.

## First-time Streamlit setup (one-time GitHub authorization by the owner)

1. Visit https://share.streamlit.io and sign in using GitHub.
2. Connect/authorize GitHub access to the public repository
   `kadhirshankar/group-sparkle-amazon-reviews`.
3. Click **Create app** > **Yup, I have an app**.
4. Set repository to `kadhirshankar/group-sparkle-amazon-reviews`,
   branch to `main`, and entrypoint file to `app.py`.
5. In **Advanced settings**, select **Python 3.14**. The repo uses Python 3.14
   and version-pinned dependencies; the Streamlit default may differ.
6. Choose a custom app subdomain if desired, and click **Deploy**.
7. After deployment, verify the left sidebar reports at least **2 / 3**
   trained models available. Enter a real review and check two live predictions.
   The third (feedforward network) is an optional local-only model.
8. Visit **Dataset & model results** and verify the five-row measured
   comparison table and macro-F1 chart.
9. Copy the confirmed public `.streamlit.app` URL into the README, shared
   final report, and presentation slides. Do not make up the URL.

## Reproducible source of the models

The app retrieves only the two files from the pinned public release:

https://github.com/kadhirshankar/group-sparkle-amazon-reviews/releases/tag/m4-seed49-verified-v1

These were produced by successful M3 Actions run 45503952149 using the official
Amazon Fine Food Reviews Kaggle dataset, the exact Group Sparkle seed 49,
the team's duplicate-removal procedure, and a stratified split.

- `multinomial_naive_bayes_bow_pipeline.joblib`
  SHA-256: `2e0df98822add52e0d74f53c0e1e3172de2b5dbe5ce395b27a6c277719f305a3`
- `logistic_regression_tfidf_pipeline.joblib`
  SHA-256: `85782818206e536afbf5e5acc71bef738c3c51a8992dd53dc393a2130ef0adcb`

The app validates each hash before loading the downloaded joblib pickle.
It does not publish private credentials, datasets, or review CSV files.
Do not load untrusted replacement model files; `joblib` uses pickle.

## Local checks

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_model_artifacts.py' -v
python -m unittest discover -s tests -p 'test_model_service.py' -v
python -m streamlit run app.py
```

The classical models auto-download on first startup. To reproduce model
training or generate the optional neural model, also run:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_data.py    # requires data/Reviews.csv from Kaggle
python scripts/train_m2_baselines.py
python scripts/train_neural_network.py
```

## Final demo checklist

- [ ] Cloud deployment URL is verified in an incognito window.
- [ ] Two trained models load with no exceptions after a cold restart.
- [ ] Positive, negative, and ambiguous reviews produce live predictions.
- [ ] Dataset/model comparison tab loads and metrics agree with team report.
- [ ] Training artifacts remain outside the committed repository.
- [ ] README, final report, and slides use the exact real URL.
- [ ] One teammate monitors the demo; another has screenshots as a fallback.

If cloud deployment fails, inspect **Manage app > Logs** in Community Cloud.
Common issues are an incorrect Python version, denied GitHub release access,
network failures, or environment-specific dependency installation issues.
