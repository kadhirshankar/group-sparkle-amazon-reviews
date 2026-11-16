"""Group Sparkle's Streamlit frontend for Amazon food review sentiment."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.model_service import ModelSpec, load_registry, predict_review


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "config" / "model_registry.json"
METRICS = ROOT / "report" / "model_metrics.json"

st.set_page_config(page_title="Group Sparkle | Sentiment Lab", page_icon="✨", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline(artifact_path: str, modified_ns: int):
    """Cache a trusted teammate-generated joblib pipeline until its file changes."""
    return joblib.load(artifact_path)


def model_states(specs: list[ModelSpec]):
    loaded = []
    unavailable = []
    for spec in specs:
        if not spec.artifact_path.is_file():
            unavailable.append((spec, "Model artifact not yet available"))
            continue
        try:
            model = load_pipeline(str(spec.artifact_path), spec.artifact_path.stat().st_mtime_ns)
            if not callable(getattr(model, "predict", None)):
                raise TypeError("Artifact does not expose predict([text])")
            loaded.append((spec, model))
        except Exception as exc:
            unavailable.append((spec, f"Unable to load model: {exc}"))
    return loaded, unavailable


try:
    specs = load_registry(REGISTRY, ROOT)
    ready, pending = model_states(specs)
    registry_error = None
except (OSError, json.JSONDecodeError, ValueError) as exc:
    specs, ready, pending = [], [], []
    registry_error = str(exc)

with st.sidebar:
    st.header("✨ Group Sparkle")
    st.caption("CSE 3521 · Amazon Fine Food Reviews")
    st.divider()
    st.metric("Models ready", f"{len(ready)} / {len(specs)}")
    for spec, _ in ready:
        st.write(f"✅ {spec.name}")
    for spec, _ in pending:
        st.write(f"⏳ {spec.name}")
    st.divider()
    st.caption("Sample seed: 49 · 50,000 reviews")
    st.caption("Predictions are produced by trained team models only.")

st.title("Amazon Review Sentiment Lab")
st.caption("Compare machine learning models on real food-review text · Group Sparkle")
if registry_error:
    st.error(f"Model configuration error: {registry_error}")

analyze_tab, comparison_tab, about_tab = st.tabs(
    ["Analyze a review", "Dataset & model results", "About the project"]
)

with analyze_tab:
    st.subheader("How does this review sound?")
    st.write("Enter a food-product review. Available trained models will classify its sentiment side by side.")
    review = st.text_area(
        "Review text",
        height=160,
        placeholder="Example: The coffee smells amazing, tastes fresh, and arrived on time.",
        help="Paste a real or invented review. The app does not save the text.",
    )
    if not ready:
        st.info(
            "The interface is ready, but trained model files have not been connected yet. "
            "Once the ML team exports its pipelines, predictions will become available."
        )
    elif len(ready) < 2:
        st.warning("Only one model is connected. The M3 demonstration requires at least two.")

    submitted = st.button("Analyze sentiment", type="primary", disabled=not ready)
    if submitted:
        if not review.strip():
            st.warning("Enter some review text first.")
        else:
            st.markdown("#### Model predictions")
            columns = st.columns(min(len(ready), 3))
            for index, (spec, pipeline) in enumerate(ready):
                with columns[index % len(columns)]:
                    with st.container(border=True):
                        st.markdown(f"**{spec.name}**")
                        st.caption(spec.feature)
                        try:
                            prediction = predict_review(pipeline, spec, review)
                            st.subheader(prediction.label)
                            if prediction.confidence is not None:
                                st.metric("Model probability", f"{prediction.confidence:.1%}")
                                st.caption("Model-estimated probability, not a guarantee.")
                            if prediction.probabilities:
                                with st.expander("Class probabilities"):
                                    for label, probability in sorted(prediction.probabilities.items()):
                                        st.write(f"{label}: {probability:.1%}")
                        except Exception as exc:
                            st.error(f"Prediction unavailable: {exc}")

    if pending:
        with st.expander("Model integration status"):
            for spec, message in pending:
                st.write(f"**{spec.name}** — {message}")
            st.caption("See docs/STREAMLIT_INTEGRATION.md for the model export contract.")

with comparison_tab:
    st.subheader("Dataset overview")
    st.caption("Class counts from the Group Sparkle M1 report (seed 49); not model predictions.")
    class_counts = pd.DataFrame(
        {"Sentiment": ["Positive", "Neutral", "Negative"], "Reviews": [39082, 3734, 7184]}
    ).set_index("Sentiment")
    st.bar_chart(class_counts)
    first, second, third = st.columns(3)
    first.metric("Sample size", "50,000")
    second.metric("Positive share", "78.16%")
    third.metric("Neutral share", "7.47%")
    st.caption("The class imbalance makes per-class and macro F1 metrics important.")
    st.divider()
    st.subheader("Model evaluation")
    if METRICS.is_file():
        try:
            metrics_data = json.loads(METRICS.read_text(encoding="utf-8"))
            rows = metrics_data.get("models", [])
            if not isinstance(rows, list) or not rows:
                raise ValueError("Expected a nonempty 'models' list")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Results supplied by the ML team on the shared held-out test set.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Could not display model metrics: {exc}")
    else:
        st.info("Evaluation metrics will appear after the ML team adds report/model_metrics.json.")

with about_tab:
    st.subheader("How this project works")
    st.write(
        "Group Sparkle compares classical machine learning with a deep/modern NLP model "
        "on a reproducible sample of Amazon Fine Food Reviews."
    )
    st.markdown(
        "**Planned feature representations:** Bag of Words and TF-IDF.  "
        "**Planned classical models:** Naive Bayes and Logistic Regression."
    )
    st.write(
        "The M1 proposal used Negative (1–2 stars), Neutral (3 stars), and Positive "
        "(4–5 stars). If the team uses binary sentiment instead, the exported model "
        "label mappings must be updated explicitly."
    )
    st.info("This app does not train models or claim results that have not been measured.")
