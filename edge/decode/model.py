"""Linear logistic trainer matching scripts.synthetic_decoder.train.

The 32-feature binary path calls that function. Wider spatial feature
vectors use a copy of the same gradient loop with n_features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.synthetic_decoder import train as train_count_decoder


def train(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Train a linear logistic model; reuse the count decoder when dim is 32."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be [examples, n_features]")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature and label counts differ")
    if x.shape[1] == 32:
        return train_count_decoder(x, y)
    return _train_n_features(x, y)


def _train_n_features(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1.0)
    x = (features - mean) / std
    w = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(600):
        z = np.clip(x @ w + bias, -30, 30)
        error = 1.0 / (1.0 + np.exp(-z)) - labels
        w -= 0.08 * (x.T @ error / len(x) + 0.001 * w)
        bias -= 0.08 * error.mean()
    raw_w = w / std
    raw_bias = float(bias - mean @ raw_w)
    return raw_w, raw_bias


def logit(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = np.asarray(features, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(bias)
    return np.clip(z, -30, 30)


def predict_labels(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return (logit(features, weights, bias) >= 0).astype(np.int8)


def logistic_confidence(score: float) -> float:
    """P(class=1) from a logit; clip for numerical stability."""
    z = float(np.clip(score, -30, 30))
    return float(1.0 / (1.0 + np.exp(-z)))
