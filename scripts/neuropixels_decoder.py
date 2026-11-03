"""Decode blank versus full right contrast from real Steinmetz Neuropixels counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.synthetic_decoder import train

SOURCE = "https://programminginneuroscience.com/wp-content/uploads/2026/09/steinmetz_session11.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/neuropixels/steinmetz_session11.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/neuropixels/decoder"))
    parser.add_argument("--report", type=Path, default=Path("reports/neuropixels-decoder.json"))
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=True) as data:
        spikes = data["spks"]
        areas = data["brain_area"]
        contrast = data["contrast_right"]
        bin_size = float(data["bin_size"])
    if spikes.shape != (698, 340, 250) or bin_size != 0.01:
        raise ValueError("unexpected session-11 shape or time bin")
    chosen_trials = np.flatnonzero((contrast == 0.0) | (contrast == 1.0))
    labels = (contrast[chosen_trials] == 1.0).astype(np.int8)
    # Fixed selection from anatomical metadata, before splitting trials.
    channels = np.flatnonzero(areas == "VISp")[:32]
    if len(channels) != 32:
        raise ValueError("fewer than 32 VISp units")
    # Stimulus appears at bin 50; decode from 50-250 ms after onset.
    counts = spikes[channels][:, chosen_trials, 55:75].sum(axis=2).T
    if np.max(counts) > 255:
        raise OverflowError("8-bit count feature saturated")
    rng = np.random.default_rng(20261103)
    train_ids = []
    test_ids = []
    for label in (0, 1):
        ids = rng.permutation(np.flatnonzero(labels == label))
        split = int(0.7 * len(ids))
        train_ids.extend(ids[:split]); test_ids.extend(ids[split:])
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)
    weights, bias = train(counts[train_ids].astype(float), labels[train_ids].astype(float))
    q_weights = np.rint(weights * 256).astype(np.int16)
    q_bias = int(np.rint(bias * 256))
    float_score = counts[test_ids] @ weights + bias
    fixed_score = counts[test_ids].astype(np.int64) @ q_weights.astype(np.int64) + q_bias
    float_prediction = (float_score >= 0).astype(np.int8)
    fixed_prediction = (fixed_score >= 0).astype(np.int8)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with (args.output / "weights.txt").open("w") as out:
        out.write(f"{q_bias}\n")
        out.writelines(f"{int(w)}\n" for w in q_weights)
    with (args.output / "rtl_vectors.txt").open("w") as out:
        for row, score in zip(counts[test_ids], fixed_score):
            packed = sum(int(value) << (8 * c) for c, value in enumerate(row))
            out.write(f"{packed:064x} {int(score)}\n")
    all_scores = counts.astype(np.int64) @ q_weights.astype(np.int64) + q_bias
    with (args.output / "all_rtl_vectors.txt").open("w") as out:
        for row, score in zip(counts, all_scores):
            packed = sum(int(value) << (8 * c) for c, value in enumerate(row))
            out.write(f"{packed:064x} {int(score)}\n")
    report = {
        "source": SOURCE,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "dataset": "Steinmetz et al. 2019 session 11, Neuromatch-derived processed trial counts",
        "task": "blank (0) versus full (1) right-side visual contrast",
        "neural_features": "first 32 VISp units, counts from 50-250 ms after stimulus onset",
        "selected_unit_indices": channels.tolist(),
        "split_seed": 20261103,
        "train_trials": len(train_ids), "test_trials": len(test_ids),
        "all_trial_rtl_stress_vectors": len(counts),
        "train_class_counts": np.bincount(labels[train_ids], minlength=2).tolist(),
        "test_class_counts": np.bincount(labels[test_ids], minlength=2).tolist(),
        "float_test_accuracy": float(np.mean(float_prediction == labels[test_ids])),
        "fixed_test_accuracy": float(np.mean(fixed_prediction == labels[test_ids])),
        "float_fixed_prediction_agreement": float(np.mean(float_prediction == fixed_prediction)),
        "majority_class_test_accuracy": float(max(np.bincount(labels[test_ids])) / len(test_ids)),
        "weight_format": "signed 16-bit Q8.8", "bias_format": "signed 32-bit Q8.8",
        "max_abs_fixed_score": int(np.max(np.abs(fixed_score))),
        "limitations": "Processed spike counts, not raw probe voltage; visual stimulus labels, not intended movement labels; one session.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(train_ids)} train / {len(test_ids)} test; float={report['float_test_accuracy']:.3f}, "
          f"fixed={report['fixed_test_accuracy']:.3f}, majority={report['majority_class_test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
