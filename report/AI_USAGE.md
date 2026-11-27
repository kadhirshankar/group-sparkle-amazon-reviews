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

## Neural network and DistilBERT (Milestone 3)

AI assistance (Claude Code) was used to install PyTorch and Hugging Face
Transformers and to help write the neural network and DistilBERT code, their
training scripts, and their tests. All results come from actually running the
code; none were made up.

## M3 Streamlit reporting improvements

AI assistance was used to connect the Streamlit comparison tab to the team's
reported M3 model metrics, add a macro-F1 chart and model-setup guidance,
and write an app handoff checklist. The metrics were taken from the
team's M3 report; no models were trained for this UI update and no
screenshots or predictions were fabricated.

## M4 deployment work

AI assistance was used to configure a versioned GitHub release for two
previously verified seed-49 classical model pipelines, implement SHA-256
verification before downloading model artifacts, split runtime from training
dependencies, and add cloud-deployment documentation and tests. No review
records were published, and no new model performance results were fabricated.
