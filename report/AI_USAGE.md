# AI assistance

AI assistance was used to set up the repository, draft documentation, and create the initial Streamlit page. No dataset analysis or model training was performed during setup.

## Streamlit application foundation

AI assistance was used to implement the initial interactive Streamlit interface,
model registry and loading adapter, integration documentation, and unit tests.
These components require team review and real trained model artifacts before
the project can make predictions. No model performance metrics were fabricated.

## Data preprocessing fixes

AI assistance was used to identify data issues in the seed-49 sample (duplicate
reviews leaking across the train/test split, HTML tags in the text, and negation
words removed by the default stop-word list) and to implement the shared
preprocessing in `src/text_preprocessing.py`. The fixes were reviewed and
verified by the team.
