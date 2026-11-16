"""Small, testable adapter between saved sklearn pipelines and Streamlit.

Model files must be produced by the ML team. This module never trains or
fabricates predictions and only loads model artifacts from the models/ folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SENTIMENTS = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    artifact_path: Path
    feature: str
    description: str
    label_map: dict[str, str]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float | None
    probabilities: dict[str, float]


def load_registry(registry_file: Path, project_root: Path) -> list[ModelSpec]:
    """Read model definitions, rejecting unsafe paths and invalid entries."""
    raw = json.loads(registry_file.read_text(encoding="utf-8"))
    entries = raw.get("models")
    if not isinstance(entries, list):
        raise ValueError("Model registry must contain a 'models' list.")

    models_dir = (project_root / "models").resolve()
    specs: list[ModelSpec] = []
    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every model registry entry must be an object.")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Each model must have a unique nonempty id.")
        identifiers.add(identifier)
        filename = entry.get("artifact", "")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Model {identifier} needs an artifact path.")
        artifact_path = (project_root / filename).resolve()
        if models_dir not in artifact_path.parents:
            raise ValueError(f"Model {identifier} artifact must be inside models/.")
        label_map = entry.get("label_map", {})
        if not isinstance(label_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.lower() in SENTIMENTS
            for key, value in label_map.items()
        ):
            raise ValueError(f"Model {identifier} has an invalid label_map.")
        specs.append(
            ModelSpec(
                id=identifier,
                name=str(entry.get("name", identifier)),
                artifact_path=artifact_path,
                feature=str(entry.get("feature", "Not specified")),
                description=str(entry.get("description", "")),
                label_map=label_map,
            )
        )
    return specs


def normalize_sentiment(raw_label: Any, label_map: Mapping[str, str]) -> str:
    """Translate the model's actual class labels without guessing numeric encodings."""
    raw_key = str(raw_label).strip()
    translated = label_map.get(raw_key, raw_key)
    result = SENTIMENTS.get(translated.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown model label {raw_key!r}. Add an explicit label_map in "
            "config/model_registry.json; binary and 3-class numeric mappings differ."
        )
    return result


def predict_review(pipeline: Any, spec: ModelSpec, review: str) -> Prediction:
    """Predict one review and optional model-estimated probabilities.

    Probabilities are shown only if the pipeline exposes predict_proba and classes_.
    They are model estimates, not externally validated confidence guarantees.
    """
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Enter a non-empty review before analyzing.")
    if not callable(getattr(pipeline, "predict", None)):
        raise TypeError(f"{spec.name} must expose predict([text]).")

    output = pipeline.predict([review])
    label = normalize_sentiment(output[0], spec.label_map)
    probabilities: dict[str, float] = {}
    if callable(getattr(pipeline, "predict_proba", None)) and hasattr(pipeline, "classes_"):
        try:
            raw_probabilities = pipeline.predict_proba([review])[0]
            classes = pipeline.classes_
            if len(classes) == len(raw_probabilities):
                for raw_class, value in zip(classes, raw_probabilities):
                    normalized = normalize_sentiment(raw_class, spec.label_map)
                    probabilities[normalized] = float(value)
        except (AttributeError, TypeError, ValueError, IndexError):
            probabilities = {}

    return Prediction(
        label=label,
        confidence=probabilities.get(label),
        probabilities=probabilities,
    )
