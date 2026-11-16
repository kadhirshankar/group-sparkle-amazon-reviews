# Streamlit App Integration — Group Sparkle

Owner: App Engineer (Kadhir Shankar)  
Status: Streamlit UI foundation implemented; trained model artifacts and measured evaluation results still needed.

## Architecture

- app.py: Streamlit interface and user interaction.
- src/model_service.py: Loads model metadata, translates class labels, and calls trained pipelines.
- config/model_registry.json: Lists the expected model artifacts and their label mappings.
- report/model_metrics.json: Optional **real** evaluation results, supplied by the ML team.

The frontend NEVER trains models or fabricates predictions.

When no trained models are present, the app shows a clear pending state and disables prediction. When one is ready, that model can predict. When two are ready, they appear side by side. Model-estimated probabilities are shown only if the fitted model provides predict_proba and classes_.

## What the machine-learning team needs to deliver

1. Use the same seed-49 sample and the agreed target-label scheme. Split into train/test **before** fitting vectorizers or classifiers. Fit feature extractors on training data only.
2. Export each fitted **scikit-learn Pipeline** containing both the text vectorizer and classifier. Each pipeline must support:

        pipeline.predict(["This snack tastes fantastic!"])

   If it supports predict_proba and classes_, the UI can additionally display model probabilities.

3. Save the trained pipelines to the artifact locations specified in config/model_registry.json. Example:

        import joblib
        joblib.dump(fitted_pipeline, "models/logistic_regression_tfidf_pipeline.joblib")

4. **Confirm binary versus three-class sentiment.** The earlier feature-engineering pull request dropped 3-star reviews, but M1 proposed three classes. The model registry must match the training labels before connecting models.

   - Literal labels such as "Positive", "Neutral", "Negative" need no mapping.
   - If binary integers mean 0 = Negative and 1 = Positive, set label_map to {"0": "Negative", "1": "Positive"} for THAT model.
   - If three-class integers mean 0 = Negative, 1 = Neutral, and 2 = Positive, use {"0": "Negative", "1": "Neutral", "2": "Positive"}.
   - These are examples only: always use the mappings that were actually used during training. The app deliberately refuses unknown integer labels.

5. Provide evaluation metrics from the shared held-out test set. An OPTIONAL file at report/model_metrics.json can use this shape:

        {
          "models": [
            {
              "model": "Logistic Regression",
              "accuracy": 0.0,
              "precision_macro": 0.0,
              "recall_macro": 0.0,
              "f1_macro": 0.0
            }
          ]
        }

   The zero values above are placeholders documenting the format, NOT real results. Do not publish fabricated metrics. Confusion matrices and 5–10 real misclassified examples belong in the team's milestone report.

## Local setup

Run from the repository root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

On Windows use .venv\Scripts\activate. The app opens even with no trained models.

To run the model adapter unit tests:

    python -m unittest discover -s tests -v

## Deployment notes for M3

- Streamlit Community Cloud: select the repository, the intended branch (usually main after review), and app.py as entrypoint.
- requirements.txt and .streamlit/config.toml are already present.
- **Deployment blocker:** the current .gitignore excludes model binaries (*.joblib) and almost everything in models/. Only local model files will not deploy to Community Cloud. Before M3, the team must choose an approved distribution mechanism, such as committing suitably small trusted trained artifacts (with a narrow .gitignore exception), Git LFS, or a controlled remote artifact source.
- Do not commit the raw Amazon reviews or the group CSV; the dataset must remain local.
- Only load trusted teammate-created joblib models; joblib uses pickle serialization and can execute arbitrary code when loading malicious files.
- After two real pipelines are connected, test the full flow and capture at least two app screenshots for M3. Add the verified deployment URL to the README.

## Next milestones

- [ ] Confirm whether sentiment classification is binary or three-class.
- [ ] Receive at least two evaluated trained pipelines and label mappings.
- [ ] Integrate real model files; test predictions and probability displays.
- [ ] Add evaluated metrics and error analysis from the ML team.
- [ ] Deploy a working app, verify it, and capture screenshots for M3.
