"""Run the trained neural network with NumPy only, so the app does not need PyTorch."""

from __future__ import annotations

import numpy as np


class NeuralSentimentPipeline:
    """TF-IDF vectorizer + trained weights. Same predict/predict_proba/classes_ as a sklearn Pipeline."""

    def __init__(self, vectorizer, weights):
        self.vectorizer = vectorizer
        self.weights = weights  # list of (W, b) NumPy arrays, one pair per Linear layer
        self.classes_ = np.array(["Negative", "Neutral", "Positive"])

    def predict_proba(self, texts):
        x = self.vectorizer.transform(texts)
        for i, (W, b) in enumerate(self.weights):
            x = x @ W + b                     # Linear layer
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)          # ReLU (no dropout: it is only used in training)
        x = np.exp(x - x.max(axis=1, keepdims=True))  # softmax: scores -> probabilities
        return x / x.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
