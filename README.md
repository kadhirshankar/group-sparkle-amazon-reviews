# Group Sparkle — Amazon Food Review Sentiment Analysis

**CSE 3521 · Dr. Thomas Torku**

We plan to compare machine learning models for classifying Amazon food reviews as positive, neutral, or negative and build a Streamlit app to display their predictions.

## Team

| Member | Role |
| --- | --- |
| Adam Spacone | |
| Alex Kull | |
| Hrushi Murukuti | |
| Kaedon Moore | |
| Parker Lemery | |
| Kadhir Shankar | App Engineer |

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

- Sample size: 50,000 reviews
- Sampling seed: 49
- Use the instructor's sampling script.
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
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`.

The app currently displays a starter page.

## Links

**Shared document:**

**Live app:**
