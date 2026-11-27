# Group Sparkle — Amazon Food Review Sentiment Analysis

**CSE 3521 · Dr. Thomas Torku**

We plan to compare machine learning models for classifying Amazon food reviews as positive, neutral, or negative and build a Streamlit app to display their predictions.

## Team

| Member | Role |
| --- | --- |
| Adam Spacone | Analyst/Writer |
| Alex Kull | Classical ML Engineer |
| Hrushi Murukuti | Project Lead |
| Kaedon Moore | Data Engineer |
| Parker Lemery | Deep Learning Engineer |
| Kadhir Shankar | App Engineer |

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

- Sample size: 50,000 reviews
- Sampling seed: 49
- Create the sample with `python scripts/prepare_data.py` (reproduces the instructor's seed-49 sample exactly from Kaggle's `Reviews.csv`).
- Dataset files are kept locally and are not uploaded to GitHub.

## Proposed Approach

- **Labels:** Negative (1–2 stars), neutral (3 stars), positive (4–5 stars)
- **Features:** Bag of Words and TF-IDF
- **Models:** Naive Bayes, Logistic Regression, and a feedforward neural network
- **App:** Streamlit

## Project Files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `data/` | Local data and instructions |
| `src/` | Project code |
| `scripts/` | Data and training scripts |
| `notebooks/` | Exploration and experiments |
| `models/` | Saved models |
| `report/` | Reports and figures |

## Run Locally

Requires Python 3.14 (see `.python-version`). The app installs only the lightweight runtime packages in `requirements.txt`. To retrain models, also install `requirements-training.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # exact versions are pinned so results match across machines
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`.

The deployed Streamlit app does **not** require PyTorch or Hugging Face Transformers. To retrain the team's classical, neural, and DistilBERT models, first run `python -m pip install -r requirements-training.txt`. Model training runs locally; the public app only loads fitted models.

For deployment, the two verified classical models automatically download from the team's [checksum-verified GitHub release](https://github.com/kadhirshankar/group-sparkle-amazon-reviews/releases/tag/m4-seed49-verified-v1). Downloaded models are checked against exact SHA-256 values in `src/model_artifacts.py` before being loaded. The optional feedforward neural model can still be trained locally using the commands below, but is not required for the deployed two-model demo. No dataset CSV is shipped in the app.

The **Dataset & model results** tab reads the team-reported M3 comparison
from `report/model_metrics.json` and shows a macro-F1 comparison chart.
The reported five-model table includes DistilBERT, which was evaluated for
the report but **is not one of the models served by this application**.
The two classical models are now available from GitHub Releases, so a fresh
checkout can make real predictions without a local Kaggle CSV. The trained
feedforward neural artifact is optional for local use only.

See [M3 app handoff and screenshot checklist](docs/M3_APP_HANDOFF.md).


## Reproduce Milestone 2

After placing the Kaggle `Reviews.csv` file in `data/`, run:

```sh
python scripts/prepare_data.py
python scripts/train_m2_baselines.py
```

The training command evaluates Naive Bayes with Bag of Words and Logistic Regression with TF-IDF against a majority-class baseline on the same held-out test set. It writes metrics, class reports, confusion matrices, informative-feature charts, and ten reproducible misclassified examples to `outputs/m2/`.

## Reproduce Milestone 3: Neural Network

After creating the sample above, run:

```sh
python scripts/train_neural_network.py
```

This trains a feedforward neural network in PyTorch on the same TF-IDF features and is tested on the same 20% test set as the baselines. 10% of the training set is held out for validation and early stopping, so the network trains on 72% of the data while the baselines train on the full 80%. The command takes about 10 seconds on a laptop CPU and writes metrics, a class report, a confusion matrix, and training/validation loss curves to `outputs/m3/`. It saves the model to `models/feedforward_nn_tfidf_pipeline.joblib`.

The saved model stores the network's weights as NumPy arrays, so the Streamlit app runs it with NumPy and does not need PyTorch installed. PyTorch is only needed for training.

## Reproduce Milestone 3: DistilBERT

```sh
caffeinate -i python scripts/train_distilbert.py   # on Mac, caffeinate keeps the computer awake; elsewhere drop it
```

This fine-tunes the pretrained `distilbert-base-uncased` model (Hugging Face Transformers) for 2 epochs on the same training, validation, and test sets as the neural network above. The first run downloads the pretrained model (about 260 MB). Training takes about 30 minutes on an Apple Silicon Mac GPU (M3 Pro) and much longer on a CPU only. Results can differ slightly between runs on a GPU, even with the same seed. Results go to `outputs/m3/` and the fine-tuned model to `models/distilbert/`. The Streamlit app does not use this model.

## Links

**Shared document:** https://docs.google.com/document/d/1HP2zxPhRQ9PAxWdidu5RLEmWWBOjoxWcxOxiRcMLlnM/edit

**Live app:** https://group-sparkle-amazon-reviews-cpxkdw3k7qspxcssk2glkh.streamlit.app/ (Streamlit Community Cloud; two trained classical models loaded in deployment screenshot, November 26, 2026).

**Deployment guide:** [docs/STREAMLIT_CLOUD_DEPLOYMENT.md](docs/STREAMLIT_CLOUD_DEPLOYMENT.md)
