"""Train and quantize a small synthetic 32-channel spike-count decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_data(seed: int = 20261102, examples: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, examples, dtype=np.int8)
    base = rng.uniform(2, 10, 32)
    modulation = np.r_[np.full(8, 0.9), np.full(8, -0.9), np.zeros(16)]
    rates = np.clip(base[None, :] + (2 * labels[:, None] - 1) * modulation[None, :], 0.2, None)
    counts = np.clip(rng.poisson(rates), 0, 255).astype(np.uint8)
    return counts, labels


def train(counts: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    # Standardize with training-set statistics only, then fold that scaling
    # into raw-count weights so RTL needs only multiply/add/compare.
    mean = counts.mean(axis=0)
    std = np.maximum(counts.std(axis=0), 1.0)
    x = (counts - mean) / std
    w = np.zeros(32, dtype=np.float64)
    bias = 0.0
    for _ in range(600):
        z = np.clip(x @ w + bias, -30, 30)
        error = 1.0 / (1.0 + np.exp(-z)) - labels
        w -= 0.08 * (x.T @ error / len(x) + 0.001 * w)
        bias -= 0.08 * error.mean()
    raw_w = w / std
    raw_bias = float(bias - mean @ raw_w)
    return raw_w, raw_bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/decoder"))
    parser.add_argument("--report", type=Path, default=Path("reports/synthetic-decoder.json"))
    args = parser.parse_args()
    counts, labels = make_data()
    train_x, test_x = counts[:3072], counts[3072:]
    train_y, test_y = labels[:3072], labels[3072:]
    weights, bias = train(train_x.astype(np.float64), train_y.astype(np.float64))
    q_weights = np.rint(weights * 256).astype(np.int16)
    q_bias = int(np.rint(bias * 256))
    fixed_score = test_x.astype(np.int64) @ q_weights.astype(np.int64) + q_bias
    float_score = test_x.astype(np.float64) @ weights + bias
    fixed_predictions = (fixed_score >= 0).astype(np.int8)
    float_predictions = (float_score >= 0).astype(np.int8)
    if np.max(np.abs(fixed_score)) > (1 << 31) - 1:
        raise OverflowError("decoder accumulator exceeds signed 32 bits")
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "test_vectors.npz", counts=test_x, labels=test_y,
             float_predictions=float_predictions, fixed_predictions=fixed_predictions,
             fixed_scores=fixed_score)
    with (args.output / "rtl_vectors.txt").open("w") as out:
        for row, score in zip(test_x, fixed_score):
            packed = sum(int(value) << (8 * i) for i, value in enumerate(row))
            out.write(f"{packed:064x} {int(score)}\n")
    with (args.output / "weights.txt").open("w") as out:
        out.write(f"{q_bias}\n")
        out.writelines(f"{int(weight)}\n" for weight in q_weights)
    report = {
        "task": "Synthetic binary latent state from 100 ms 32-channel Poisson spike-count windows",
        "label_meaning": {"0": "synthetic state A", "1": "synthetic state B"},
        "seed": 20261102, "train_examples": 3072, "test_examples": 1024,
        "features": "32 unsigned 8-bit per-channel counts; first 16 channels carry opposing state modulation",
        "float_test_accuracy": float(np.mean(float_predictions == test_y)),
        "fixed_test_accuracy": float(np.mean(fixed_predictions == test_y)),
        "float_fixed_prediction_agreement": float(np.mean(float_predictions == fixed_predictions)),
        "weight_format": "signed Q8.8 in 16 bits", "bias_format": "signed Q8.8 in 32 bits",
        "accumulator_format": "signed 32-bit integer in Q8.8 units",
        "maximum_absolute_fixed_score": int(np.max(np.abs(fixed_score))),
        "limitation": "Labels are generated from a synthetic rate model; this is not movement decoding or real Neuropixels accuracy.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Float accuracy {report['float_test_accuracy']:.3f}; fixed accuracy {report['fixed_test_accuracy']:.3f}; "
          f"agreement {report['float_fixed_prediction_agreement']:.3f}")


if __name__ == "__main__":
    main()
