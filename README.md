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

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # exact versions are pinned so results match across machines
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`.

The app loads the trained models listed in `config/model_registry.json`. Run the Milestone 2 commands below first so the model files exist in `models/`.

## Reproduce Milestone 2

After placing the Kaggle `Reviews.csv` file in `data/`, run:

```sh
python scripts/prepare_data.py
python scripts/train_m2_baselines.py
```

The training command evaluates Naive Bayes with Bag of Words and Logistic Regression with TF-IDF against a majority-class baseline on the same held-out test set. It writes metrics, class reports, confusion matrices, informative-feature charts, and ten reproducible misclassified examples to `outputs/m2/`.

## Links

**Shared document:**

**Live app:**
