"""Compare 32-bin spike-count baseline vs spatially concatenated features."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

for _thread_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import numpy as np

from edge.decode.features import N_CHANNELS, N_Y_BINS, SPATIAL_DIM
from edge.decode.model import predict_labels, train
from edge.decode.spatial_task import (
    FORWARD_MODEL,
    JITTER_SIGMA_UM,
    LOCALIZER_FALLBACK,
    N_TEST,
    N_TRAIN,
    WINDOW_S,
    feature_matrices,
    generate_dataset,
    source_table,
    split_indices,
)

NEUROPIXELS_ZIP = Path("data/neuropixels/steinmetz_session11.zip")
NEUROPIXELS_SOURCE = "https://programminginneuroscience.com/wp-content/uploads/2026/09/steinmetz_session11.zip"


def run_synthetic_comparison(
    seed: int = 20261103,
    examples: int | None = None,
    n_train: int | None = None,
) -> dict[str, object]:
    n_examples = N_TRAIN + N_TEST if examples is None else int(examples)
    n_tr = N_TRAIN if n_train is None else int(n_train)
    if examples is not None and n_train is None:
        n_tr = int(0.75 * n_examples)
    dataset = generate_dataset(seed=seed, examples=n_examples)
    baseline_x, spatial_x = feature_matrices(dataset)
    labels = dataset.labels
    train_idx, test_idx = split_indices(len(labels), n_tr)
    train_y = labels[train_idx].astype(np.float64)
    test_y = labels[test_idx]

    base_w, base_b = train(baseline_x[train_idx], train_y)
    spatial_w, spatial_b = train(spatial_x[train_idx], train_y)
    base_pred = predict_labels(baseline_x[test_idx], base_w, base_b)
    spatial_pred = predict_labels(spatial_x[test_idx], spatial_w, spatial_b)
    base_acc = float(np.mean(base_pred == test_y))
    spatial_acc = float(np.mean(spatial_pred == test_y))
    majority = float(max(np.bincount(test_y, minlength=2)) / len(test_y))
    improved = bool(spatial_acc > base_acc)

    return {
        "task": (
            "Synthetic binary spatial-cluster identity: which of two neighboring "
            f"source clusters along probe y is more active in a {WINDOW_S * 1000:.0f} ms "
            "window, with label-independent tonic sources along the shank"
        ),
        "label_meaning": {
            "0": "deep cluster (low y_um) more active",
            "1": "shallow cluster (high y_um) more active",
        },
        "n_classes": 2,
        "seed": seed,
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "split": "contiguous first-N train after seeded generation; same indices for both models",
        "sources": source_table(dataset.sources),
        "forward_model": FORWARD_MODEL,
        "used_stub_forward": True,  # simplified monopole, not a missing renderer
        "localizer_method": dataset.localizer_method,
        "localization_fallback": LOCALIZER_FALLBACK,
        "jitter_sigma_um": JITTER_SIGMA_UM,
        "window_s": WINDOW_S,
        "n_y_bins": N_Y_BINS,
        "majority_class_test_accuracy": majority,
        "baseline": {
            "features": "32 unsigned per-channel spike counts from forward-model peak channels",
            "n_features": N_CHANNELS,
            "trainer": "scripts.synthetic_decoder.train",
            "test_accuracy": base_acc,
        },
        "spatial": {
            "features": (
                f"{N_CHANNELS} counts concatenated with {N_Y_BINS} probe-y occupancy "
                "bins and mean estimated xyz_um"
            ),
            "n_features": SPATIAL_DIM,
            "trainer": "edge.decode.model.train (copied gradient loop, n_features != 32)",
            "test_accuracy": spatial_acc,
        },
        "spatial_improved": improved,
        "accuracy_delta_spatial_minus_baseline": float(spatial_acc - base_acc),
        "same_split": True,
    }


def run_neuropixels_count_only(zip_path: Path = NEUROPIXELS_ZIP) -> dict[str, object]:
    """Count-only protocol from scripts/neuropixels_decoder.py; spatial features off."""
    if not zip_path.exists():
        return {
            "status": "skipped",
            "spatial_features": "off",
            "zip_path": str(zip_path),
            "limitation": (
                "Steinmetz session-11 zip is not present. Neuropixels path skipped. "
                "Fixture sorted-spike playback is not voltage and was not used as ADC input."
            ),
        }

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.synthetic_decoder import train as train32

    with np.load(zip_path, allow_pickle=True) as data:
        spikes = data["spks"]
        areas = data["brain_area"]
        contrast = data["contrast_right"]
        bin_size = float(data["bin_size"])
    if spikes.shape != (698, 340, 250) or bin_size != 0.01:
        raise ValueError("unexpected session-11 shape or time bin")
    chosen_trials = np.flatnonzero((contrast == 0.0) | (contrast == 1.0))
    labels = (contrast[chosen_trials] == 1.0).astype(np.int8)
    channels = np.flatnonzero(areas == "VISp")[:32]
    if len(channels) != 32:
        raise ValueError("fewer than 32 VISp units")
    counts = spikes[channels][:, chosen_trials, 55:75].sum(axis=2).T
    rng = np.random.default_rng(20261103)
    train_ids: list[int] = []
    test_ids: list[int] = []
    for label in (0, 1):
        ids = rng.permutation(np.flatnonzero(labels == label))
        split = int(0.7 * len(ids))
        train_ids.extend(ids[:split])
        test_ids.extend(ids[split:])
    train_ids_arr = np.array(train_ids)
    test_ids_arr = np.array(test_ids)
    weights, bias = train32(counts[train_ids_arr].astype(float), labels[train_ids_arr].astype(float))
    float_score = counts[test_ids_arr] @ weights + bias
    float_prediction = (float_score >= 0).astype(np.int8)
    acc = float(np.mean(float_prediction == labels[test_ids_arr]))
    return {
        "status": "ran_count_only",
        "spatial_features": "off",
        "source": NEUROPIXELS_SOURCE,
        "source_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "task": "blank (0) versus full (1) right-side visual contrast",
        "neural_features": "first 32 VISp units, counts from 50-250 ms after stimulus onset",
        "split_seed": 20261103,
        "train_trials": int(len(train_ids_arr)),
        "test_trials": int(len(test_ids_arr)),
        "float_test_accuracy": acc,
        "trainer": "scripts.synthetic_decoder.train",
        "limitation": (
            "Processed spike counts, not raw probe voltage. Spatial localization "
            "features are off because this extract has no per-spike xyz in probe_um. "
            "Visual stimulus labels are not perceptions or intended movement. "
            "Fixture sorted spikes were not treated as voltages."
        ),
    }


def build_report(seed: int = 20261103, zip_path: Path = NEUROPIXELS_ZIP) -> dict[str, object]:
    synthetic = run_synthetic_comparison(seed=seed)
    neuropixels = run_neuropixels_count_only(zip_path=zip_path)
    return {
        **synthetic,
        "neuropixels": neuropixels,
        "rtl_count_baseline": (
            "rtl/spike_count_decoder.sv remains the 32-count RTL baseline and was not modified"
        ),
        "limitations": [
            "Synthetic labels mark which simulated source cluster was rate-modulated; they are not perceptions.",
            "This experiment is not vision restoration, not intended-movement decoding, and not a claim about humans.",
            FORWARD_MODEL,
            "Localization uses an importable LocalizationEngine when present; otherwise ground-truth xyz plus Gaussian jitter.",
            "Spatial improvement is reported only if held-out accuracy rose on the same split; it is not assumed.",
            "Neuropixels CCF coordinates are not human neuron coordinates and were not mapped through edge.coords.",
        ],
        "n_test_expected": N_TEST,
    }
