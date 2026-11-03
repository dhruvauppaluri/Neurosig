"""Leakage-free three-way spatial decoding comparison on held-out synthetic trials."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from edge.decode.features import N_CHANNELS, SPATIAL_DIM, stack_spatial
from edge.decode.model import predict_labels, train
from edge.decode.spatial_task import generate_dataset, split_indices
from edge.geometry import linear_probe_um


def classification(y: np.ndarray, pred: np.ndarray) -> dict:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, guess in zip(y, pred):
        matrix[int(truth), int(guess)] += 1
    return {
        "accuracy": float(np.mean(y == pred)), "confusion_matrix": matrix.tolist(),
        "precision": [float(matrix[c, c] / max(1, matrix[:, c].sum())) for c in (0, 1)],
        "recall": [float(matrix[c, c] / max(1, matrix[c, :].sum())) for c in (0, 1)],
    }


def _channel_positions(dataset) -> list[np.ndarray]:
    xyz = linear_probe_um().xyz_um()
    return [xyz[channels] for channels in dataset.spike_channels]


def run(seeds: tuple[int, ...] = (20261103, 20261104, 20261105),
        examples: int = 1024) -> dict:
    seed_reports = []
    for seed in seeds:
        dataset = generate_dataset(seed=seed, examples=examples)
        counts = dataset.counts.astype(float)
        inputs = {
            "A_counts": counts,
            "B_electrode": stack_spatial(counts, _channel_positions(dataset)),
            "C_estimated": stack_spatial(counts, dataset.estimated_xyz),
        }
        train_idx, test_idx = split_indices(examples, int(0.75 * examples))
        y_train, y_test = dataset.labels[train_idx], dataset.labels[test_idx]
        models = {}
        for name, x in inputs.items():
            weights, bias = train(x[train_idx], y_train.astype(float))
            start = time.perf_counter()
            pred = predict_labels(x[test_idx], weights, bias)
            elapsed = time.perf_counter() - start
            models[name] = {**classification(y_test, pred), "n_features": int(x.shape[1]),
                            "model_bytes_float64": int(weights.nbytes + 8),
                            "inference_us_per_trial": elapsed * 1e6 / len(test_idx)}
        # Perturb only estimates, never known source positions. A fixed model
        # makes this a test-time robustness sweep rather than retraining.
        c_train = inputs["C_estimated"][train_idx]
        c_w, c_b = train(c_train, y_train.astype(float))
        sensitivity = {}
        for sigma in (0.0, 20.0, 50.0, 100.0):
            rng = np.random.default_rng(seed + int(sigma) + 991)
            estimated = []
            for points in dataset.estimated_xyz:
                estimated.append(points + rng.normal(0.0, sigma, size=points.shape))
            x_test = stack_spatial(counts[test_idx], [estimated[i] for i in test_idx])
            sensitivity[str(int(sigma))] = classification(y_test, predict_labels(x_test, c_w, c_b))["accuracy"]
        seed_reports.append({"seed": seed, "train": len(train_idx), "test": len(test_idx),
                             "localizer": dataset.localizer_method,
                             "missing_estimate_fallback": dataset.localizer_fallback,
                             "models": models, "estimate_perturbation_um": sensitivity})
    aggregate = {}
    for name in ("A_counts", "B_electrode", "C_estimated"):
        accuracies = [r["models"][name]["accuracy"] for r in seed_reports]
        aggregate[name] = {"mean_accuracy": float(np.mean(accuracies)),
                           "min_accuracy": float(np.min(accuracies)),
                           "max_accuracy": float(np.max(accuracies)),
                           "std_accuracy": float(np.std(accuracies))}
    return {"task": "synthetic spatial-cluster activity", "forward_model": "simplified monopole",
            "feature_policy": "A counts; B counts plus geometry of observed peak channels; C counts plus waveform-localizer estimates, falling back to peak-channel geometry only on failure",
            "ground_truth_feature_leakage": False, "n_channels": N_CHANNELS,
            "spatial_feature_width": SPATIAL_DIM, "seeds": seed_reports, "aggregate": aggregate,
            "rtl_baseline": "existing 32-count RTL decoder is a separately trained task baseline; no RTL inference claimed here"}


def main() -> None:
    report = run()
    target = Path("reports/edge-decode-v2.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
