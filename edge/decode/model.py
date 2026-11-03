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


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    hidden: int = 16,
    seed: int = 0,
    steps: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tiny one-hidden-layer logistic net. Edge-computer only; not RTL."""
    rng = np.random.default_rng(seed)
    x_raw = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1.0)
    x = (x_raw - mean) / std
    w1 = rng.normal(0.0, 0.15, size=(x.shape[1], hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(0.0, 0.15, size=(hidden,))
    b2 = 0.0
    rate = 0.08
    for _ in range(int(steps)):
        hidden_pre = x @ w1 + b1
        hidden_act = np.tanh(hidden_pre)
        z = np.clip(hidden_act @ w2 + b2, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-z))
        error = pred - y
        d_w2 = hidden_act.T @ error / len(x) + 0.001 * w2
        d_b2 = float(error.mean())
        d_hidden = (error[:, None] * w2[None, :]) * (1.0 - np.square(hidden_act))
        d_w1 = x.T @ d_hidden / len(x) + 0.001 * w1
        d_b1 = d_hidden.mean(axis=0)
        w2 -= rate * d_w2
        b2 -= rate * d_b2
        w1 -= rate * d_w1
        b1 -= rate * d_b1
    return w1 / std[:, None], b1 - (mean / std) @ (w1), w2, float(b2)


def predict_mlp(
    features: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> np.ndarray:
    hidden_act = np.tanh(np.asarray(features, dtype=np.float64) @ w1 + b1)
    z = np.clip(hidden_act @ w2 + b2, -30, 30)
    return (z >= 0).astype(np.int8)


def mcnemar_pvalue(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact binomial McNemar test on paired correctness. Two-sided."""
    truth = np.asarray(y)
    a = np.asarray(pred_a) == truth
    b = np.asarray(pred_b) == truth
    n01 = int(np.count_nonzero(a & ~b))
    n10 = int(np.count_nonzero(~a & b))
    n = n01 + n10
    if n == 0:
        pvalue = 1.0
    else:
        k = min(n01, n10)
        # two-sided exact binomial with p=0.5
        from math import comb
        cdf = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        pvalue = min(1.0, 2.0 * cdf)
    return {"n01_a_only": n01, "n10_b_only": n10, "n_discordant": n, "p_two_sided": float(pvalue)}


def bootstrap_accuracy_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    seed: int,
    n_boot: int = 400,
) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a = np.asarray(pred_a) == y
    b = np.asarray(pred_b) == y
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    delta = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "mean_delta": float(np.mean(delta)),
        "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "n_boot": n_boot,
    }


def logistic_confidence(score: float) -> float:
    """P(class=1) from a logit; clip for numerical stability."""
    z = float(np.clip(score, -30, 30))
    return float(1.0 / (1.0 + np.exp(-z)))
