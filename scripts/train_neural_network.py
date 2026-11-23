"""Train the feedforward neural network (PyTorch) on TF-IDF features."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_model import FeedforwardNet  # noqa: E402
from src.neural_pipeline import NeuralSentimentPipeline  # noqa: E402
from src.text_preprocessing import load_and_split, make_vectorizer  # noqa: E402
from src.training_utils import LABELS, calculate_metrics, save_confusion_matrix  # noqa: E402


def prepare_data(data_path):
    """Same 80/20 split as the baselines, plus 10% of train held out for validation."""
    train, test = load_and_split(data_path)
    train, val = train_test_split(train, test_size=0.1, stratify=train["sentiment"], random_state=49)
    return train, val, test


def to_ints(df):
    """"Negative"/"Neutral"/"Positive" -> 0/1/2"""
    return np.array([LABELS.index(label) for label in df["sentiment"]])


def batches(X, y, batch_size=256, shuffle=False):
    """Yield (features, labels) tensors one small batch at a time."""
    order = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
    for start in range(0, len(y), batch_size):
        rows = order[start:start + batch_size]
        features = torch.tensor(X[rows].toarray(), dtype=torch.float32)  # sparse -> dense, this batch only
        yield features, torch.tensor(y[rows])


def evaluate(model, X, y, loss_fn):
    """Average loss and predicted class numbers, with dropout off and no gradients."""
    model.eval()
    total_loss, predictions = 0.0, []
    with torch.no_grad():
        for features, labels in batches(X, y):
            logits = model(features)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            predictions.append(logits.argmax(dim=1).numpy())
    return total_loss / len(y), np.concatenate(predictions)


def train_model(train_data, val_data, hidden1=256, hidden2=64, dropout=0.3, lr=1e-3, max_epochs=20, patience=3):
    """Train with class-weighted loss; keep the epoch with the best validation macro F1."""
    torch.manual_seed(49)
    np.random.seed(49)
    X_train, y_train = train_data
    X_val, y_val = val_data

    model = FeedforwardNet(X_train.shape[1], hidden1, hidden2, dropout)
    counts = np.bincount(y_train)
    class_weights = len(y_train) / (3 * counts)  # rare classes count more, like class_weight="balanced"
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in batches(X_train, y_train, shuffle=True):
            logits = model(features)           # 1. forward pass
            loss = loss_fn(logits, labels)     # 2. how wrong were we?
            optimizer.zero_grad()              # 3. clear last batch's gradients
            loss.backward()                    # 4. backpropagation
            optimizer.step()                   # 5. update the weights
            total_loss += loss.item() * len(labels)

        val_loss, val_predictions = evaluate(model, X_val, y_val, loss_fn)
        val_f1 = f1_score(y_val, val_predictions, average="macro")
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_train), "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch {epoch:2d}  train loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  val macro F1 {val_f1:.4f}")

        if val_f1 > best_f1:  # early stopping: remember the best epoch, stop after `patience` epochs with no gain
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def save_loss_curves(history, output_dir):
    """Training vs validation loss (left) and validation macro F1 (right), per epoch."""
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "feedforward_nn_tfidf_history.csv", index=False)
    best = df.loc[df["val_macro_f1"].idxmax(), "epoch"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    left.plot(df["epoch"], df["train_loss"], "o-", label="Training loss")
    left.plot(df["epoch"], df["val_loss"], "o-", label="Validation loss")
    left.set(title="Loss per epoch", xlabel="Epoch", ylabel="Class-weighted cross-entropy")
    right.plot(df["epoch"], df["val_macro_f1"], "o-", color="tab:green")
    right.set(title="Validation macro F1 per epoch", xlabel="Epoch", ylabel="Macro F1")
    for ax in (left, right):
        ax.set_xticks(df["epoch"])
        ax.axvline(best, color="gray", linestyle="--", label=f"Kept epoch ({best})")
        ax.legend()
    fig.suptitle("Feedforward Neural Network (TF-IDF)")
    fig.tight_layout()
    fig.savefig(output_dir / "feedforward_nn_tfidf_loss_curves.png", dpi=200)
    plt.close(fig)


def export_pipeline(model, vectorizer, X_test, test_texts):
    """Copy the weights out of PyTorch into NumPy and check both give the same probabilities."""
    weights = [
        (layer.weight.detach().numpy().T.copy(), layer.bias.detach().numpy().copy())
        for layer in model.layers if isinstance(layer, nn.Linear)
    ]
    pipeline = NeuralSentimentPipeline(vectorizer, weights)

    model.eval()
    with torch.no_grad():
        torch_probs = torch.softmax(model(torch.tensor(X_test[:500].toarray(), dtype=torch.float32)), dim=1).numpy()
    numpy_probs = pipeline.predict_proba(test_texts[:500])
    print(f"NumPy check: max difference from PyTorch = {np.abs(torch_probs - numpy_probs).max():.2e}")
    assert np.allclose(torch_probs, numpy_probs, atol=1e-5), "NumPy pipeline does not match PyTorch"
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/group_sample_seed49.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m3"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "feedforward_nn_tfidf"

    train, val, test = prepare_data(args.data)

    # timed like the baselines' pipeline.fit(): fitting TF-IDF + training the model
    started = time.perf_counter()
    vectorizer = make_vectorizer("tfidf")
    X_train = vectorizer.fit_transform(train["Text"])  # vocabulary learned from training reviews only
    X_val = vectorizer.transform(val["Text"])
    model, history = train_model((X_train, to_ints(train)), (X_val, to_ints(val)))
    seconds = time.perf_counter() - started

    X_test, y_test, test_texts = vectorizer.transform(test["Text"]), to_ints(test), test["Text"]

    _, test_predictions = evaluate(model, X_test, y_test, nn.CrossEntropyLoss())
    y_true = np.array(LABELS)[y_test]  # back to "Negative"/"Neutral"/"Positive" to match the baselines
    y_pred = np.array(LABELS)[test_predictions]

    metrics = calculate_metrics(name, y_true, y_pred, seconds)
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"{name}_metrics.csv", index=False)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(args.output_dir / f"{name}_class_report.csv")
    save_confusion_matrix(name, y_true, y_pred, args.output_dir,
                          title="Feedforward Neural Network (TF-IDF) Confusion Matrix")
    save_loss_curves(history, args.output_dir)

    pipeline = export_pipeline(model, vectorizer, X_test, test_texts)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.models_dir / f"{name}_pipeline.joblib")

    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Artifacts written to {args.output_dir}; model saved to {args.models_dir}")


if __name__ == "__main__":
    main()
