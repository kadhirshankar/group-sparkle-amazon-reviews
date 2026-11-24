"""Shared reporting helpers for the classical sentiment models."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

LABELS = ["Negative", "Neutral", "Positive"]


def calculate_metrics(name, y_true, y_pred, seconds):
    """Return overall, macro, and weighted metrics for one model."""
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
        "training_seconds": seconds,
    }


def save_confusion_matrix(name, y_true, y_pred, output_dir: Path, title=None) -> None:
    """Save a confusion matrix as both CSV and a report-ready PNG."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        output_dir / f"{name}_confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(
        title=title or f"{name.replace('_', ' ').title()} Confusion Matrix",
        xlabel="Predicted sentiment",
        ylabel="Actual sentiment",
    )
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS, rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def train_and_evaluate(
    name, pipeline: Pipeline, train, test, output_dir: Path, model_path: Path | None = None
):
    """Fit one pipeline and write its required M2 evaluation artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipeline.fit(train["Text"], train["sentiment"])
    elapsed = time.perf_counter() - started
    predictions = pipeline.predict(test["Text"])
    report = classification_report(
        test["sentiment"], predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{name}_class_report.csv")
    save_confusion_matrix(name, test["sentiment"].to_numpy(), predictions, output_dir)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
    return (
        calculate_metrics(name, test["sentiment"].to_numpy(), predictions, elapsed),
        predictions,
    )
