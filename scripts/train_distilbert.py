"""Fine-tune pretrained DistilBERT (Hugging Face) on the review sentiment labels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_neural_network import prepare_data, save_loss_curves, to_ints  # noqa: E402
from src.text_preprocessing import clean_text  # noqa: E402
from src.training_utils import LABELS, calculate_metrics, save_confusion_matrix  # noqa: E402

MODEL_NAME = "distilbert-base-uncased"
MAX_TOKENS = 256  # 94% of training reviews fit completely; longer ones are cut
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU if available


def batches(texts, labels, batch_size=16, shuffle=False):
    """Yield (tokens, labels) one batch at a time, padded only to the longest review in the batch."""
    texts = [clean_text(t) for t in texts]  # same HTML removal / "isn't" -> "is not" as the other models
    order = np.random.permutation(len(labels)) if shuffle else np.arange(len(labels))
    for start in range(0, len(labels), batch_size):
        rows = order[start:start + batch_size]
        tokens = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                           max_length=MAX_TOKENS, return_tensors="pt")
        yield tokens, torch.tensor(labels[rows])


def evaluate(model, texts, labels, loss_fn):
    """Average loss and predicted class numbers, with dropout off and no gradients."""
    model.eval()
    total_loss, predictions = 0.0, []
    with torch.no_grad():
        for tokens, y in batches(texts, labels, batch_size=64):
            logits = model(**tokens.to(DEVICE)).logits
            total_loss += loss_fn(logits, y.to(DEVICE)).item() * len(y)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return total_loss / len(labels), np.concatenate(predictions)


def train_model(train_texts, y_train, val_texts, y_val, lr=2e-5, epochs=2):
    """Fine-tune all of DistilBERT; keep the epoch with the best validation macro F1."""
    torch.manual_seed(49)
    np.random.seed(49)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
    class_weights = len(y_train) / (3 * np.bincount(y_train))  # same as Model A and the baselines
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (tokens, y) in enumerate(batches(train_texts, y_train, shuffle=True), start=1):
            y = y.to(DEVICE)
            logits = model(**tokens.to(DEVICE)).logits  # 1. forward pass
            loss = loss_fn(logits, y)                    # 2. how wrong were we?
            optimizer.zero_grad()                        # 3. clear last batch's gradients
            loss.backward()                              # 4. backpropagation through all 6 layers
            optimizer.step()                             # 5. update the weights
            total_loss += loss.item() * len(y)
            if step % 200 == 0:
                print(f"  epoch {epoch}  batch {step}  running loss {total_loss / (step * 16):.4f}", flush=True)

        val_loss, val_predictions = evaluate(model, val_texts, y_val, loss_fn)
        val_f1 = f1_score(y_val, val_predictions, average="macro")
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_train), "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch {epoch}  train loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  val macro F1 {val_f1:.4f}", flush=True)

        if val_f1 > best_f1:  # keep a CPU copy of the best epoch's weights
            best_f1 = val_f1
            best_state = {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/group_sample_seed49.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m3"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "distilbert"

    train, val, test = prepare_data(args.data)  # same split and validation set as Model A
    started = time.perf_counter()
    model, history = train_model(train["Text"].tolist(), to_ints(train), val["Text"].tolist(), to_ints(val))
    seconds = time.perf_counter() - started

    _, test_predictions = evaluate(model, test["Text"].tolist(), to_ints(test), nn.CrossEntropyLoss())
    y_true = np.array(LABELS)[to_ints(test)]
    y_pred = np.array(LABELS)[test_predictions]

    metrics = calculate_metrics(name, y_true, y_pred, seconds)
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"{name}_metrics.csv", index=False)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(args.output_dir / f"{name}_class_report.csv")
    save_confusion_matrix(name, y_true, y_pred, args.output_dir, title="DistilBERT (fine-tuned) Confusion Matrix")
    save_loss_curves(history, args.output_dir, name, "DistilBERT (fine-tuned)")

    model.save_pretrained(args.models_dir / name)  # ~260 MB; not used by the app
    tokenizer.save_pretrained(args.models_dir / name)

    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Artifacts written to {args.output_dir}; model saved to {args.models_dir / name}")


if __name__ == "__main__":
    main()
