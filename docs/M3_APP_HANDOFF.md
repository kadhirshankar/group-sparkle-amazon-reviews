# Milestone 3 — Streamlit App Handoff (App Engineer)

This file is a factual implementation checklist for the team's own M3 report writing.

## Current architecture

- `app.py` is the Streamlit entrypoint.
- `src/model_service.py` loads model metadata, normalizes labels, and runs predictions.
- `config/model_registry.json` lists three supported artifact files (Naive Bayes/BoW, Logistic Regression/TF-IDF, feedforward neural network/TF-IDF).
- `report/model_metrics.json` contains team-reported held-out results from the shared M3 document, so the dashboard can show the full model comparison even if a trained artifact is missing.
- DistilBERT is evaluated in the M3 report but **is not integrated into the Streamlit runtime**, because its saved artifact is large.
- `models/*.joblib` files are excluded from Git and must be generated or supplied locally. Merely registering a model does **not** make it available for predictions.

## App tabs and interaction (facts to check and describe in your own words)

1. **Analyze a review:** type or paste review text; click *Analyze sentiment*. For each loaded model, the app shows Positive/Neutral/Negative and model-estimated probabilities (when supported). When model artifacts are missing, a visible warning explains why predictions are disabled.
2. **Dataset & model results:** displays the seed-49 class distribution (50,000 reviews), class imbalance notes, and team-measured model comparison metrics. The updated app also charts macro F1.
3. **About the project:** explains the feature representations, sentiment label mapping, and the distinction between trained model results and the UI.

**Do not claim that predictions work until the saved artifacts have been generated and real input/output verified.**

## Run locally and check real predictions

From the repository root, on a machine with the required libraries and dataset:

```sh
python scripts/prepare_data.py
python scripts/train_m2_baselines.py
python scripts/train_neural_network.py
python -m streamlit run app.py
```

The three models should be reported as ready if their artifacts load successfully; the app should accept a review and produce live predictions. Do not run the costly DistilBERT training command just to use the app.

If a teammate already trained the models, copy only the **trusted** files into `models/` using the exact artifact names in `config/model_registry.json`. `joblib` is pickle-based: never load a model file from an untrusted source.

## Screenshots required for the M3 PDF

The instructor requires **at least two screenshots of the real running app**:

1. **Screenshot 1 — Review analyzer:** run the application with at least two real trained model files and enter a food review. Capture the text field and side-by-side non-mock model predictions.
2. **Screenshot 2 — Dataset & model results:** capture the chart of the class distribution, real five-row model comparison, and macro-F1 comparison chart.

Capture screenshots from the running Streamlit session. **No design mockup or static screenshot is a substitute.** Insert both in the standalone 4–6-page M3 PDF. Keep text concise to make room.

## Deployment

M3 allows a **local running app or staging deployment**. A deployed Streamlit Community Cloud application is required for the final project.

For Community Cloud, the hosted code needs trained artifacts, even if the original data stays local. Before deployment, decide with the ML engineers whether to version small trusted serialized pipelines in the repo, use a trusted release artifact, or another controlled distribution mechanism. Training files and dependency pins may need further environment review. Do not claim a live URL until it is verified.

## Individual progress log (App Engineer)

**Current (adapt in your own words after checking):** Completed and merged the Streamlit UI foundation, model-loading adapter, and tests. Added the M3 comparison dashboard using the team's measured results, and documented how to run and capture the application.

**Next:** Verify the three trained pipelines locally, collect genuine screenshots, finish the deployment path, and assist with the final live demo.
