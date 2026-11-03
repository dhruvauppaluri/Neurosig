"""Decode visual-stimulus labels from frozen-detector events, not only sorting.

Streams a contiguous native AP block covering eligible trials, runs the frozen
Python detector, and bins detections in the same 50–250 ms post-stimulus window
used by the sorted-count decoder. A second public insertion is used for
cross-recording tests. Sorted counts remain a comparison, not ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edge.decode.experiment_v2 import classification
from edge.decode.model import bootstrap_accuracy_delta, mcnemar_pvalue, predict_labels, train
from edge.ibl import DEFAULT_PID, SECOND_PID, SORTING_REVISION, TRIALS_REVISION, load_raw_chunk


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
from scripts.real_detector import (
    FROZEN_REAL_DETECTOR_NAME,
    FROZEN_REAL_DETECTOR_RATIONALE,
    frozen_real_detector,
    detect_real,
)


def _eligible_trials(trials) -> tuple[np.ndarray, np.ndarray]:
    left = np.nan_to_num(np.asarray(trials["contrastLeft"], dtype=float), nan=0.0)
    right = np.nan_to_num(np.asarray(trials["contrastRight"], dtype=float), nan=0.0)
    onset = np.asarray(trials["stimOn_times"], dtype=float)
    eligible = np.flatnonzero(np.isfinite(onset) & ((left > 0) ^ (right > 0)))
    labels = (right[eligible] > 0).astype(np.int8)
    return onset[eligible], labels


def _counts_from_events(
    event_samples: np.ndarray,
    event_channels: np.ndarray,
    stim_times: np.ndarray,
    *,
    t0_s: float,
    fs: float,
    n_channels: int,
    start_s: float = 0.05,
    stop_s: float = 0.25,
) -> np.ndarray:
    counts = np.zeros((len(stim_times), n_channels), dtype=float)
    for row, stim in enumerate(stim_times):
        first = (stim - t0_s + start_s) * fs
        last = (stim - t0_s + stop_s) * fs
        mask = (event_samples >= first) & (event_samples < last)
        for channel in event_channels[mask]:
            if 0 <= int(channel) < n_channels:
                counts[row, int(channel)] += 1.0
    return counts


def _sorted_counts(spike_times, spike_clusters, selected_clusters, stim_times) -> np.ndarray:
    counts = np.zeros((len(stim_times), len(selected_clusters)), dtype=float)
    cluster_index = {int(cluster): i for i, cluster in enumerate(selected_clusters)}
    for row, stim in enumerate(stim_times):
        start = int(np.searchsorted(spike_times, stim + 0.05))
        stop = int(np.searchsorted(spike_times, stim + 0.25))
        for cluster in spike_clusters[start:stop]:
            index = cluster_index.get(int(cluster))
            if index is not None:
                counts[row, index] += 1.0
    return counts


def _session_block(pid: str, t0_s: float, duration_s: float) -> dict:
    from one.api import ONE

    chunk = load_raw_chunk(pid, t0_s, duration_s)
    one = ONE(base_url="https://openalyx.internationalbrainlab.org",
              username="intbrainlab", password="international", silent=True,
              cache_dir="data/neuropixels/ibl_cache")
    trials = one.load_dataset(chunk["eid"], "_ibl_trials.table.pqt", collection="alf",
                              revision=TRIALS_REVISION)
    onsets, labels = _eligible_trials(trials)
    in_block = (onsets >= t0_s + 0.05) & (onsets < t0_s + duration_s - 0.25)
    onsets, labels = onsets[in_block], labels[in_block]
    detected = detect_real(chunk["voltages_uv"], frozen_real_detector(chunk["fs"]))
    events = detected["events"]
    event_samples = np.array([event.sample for event in events], dtype=float)
    event_channels = np.array([event.channel for event in events], dtype=int)
    detector_counts = _counts_from_events(
        event_samples, event_channels, onsets, t0_s=t0_s, fs=chunk["fs"],
        n_channels=chunk["voltages_uv"].shape[1],
    )
    selected_clusters = np.unique(chunk["spike_clusters"][chunk["sorter_mask"]])[:32]
    if len(selected_clusters) < 32:
        selected_clusters = np.unique(chunk["spike_clusters"])[:32]
    sorted_counts = _sorted_counts(
        chunk["spike_times"], chunk["spike_clusters"], selected_clusters, onsets
    )
    return {
        "pid": pid,
        "eid": str(chunk["eid"]),
        "t0_s": t0_s,
        "duration_s": duration_s,
        "n_trials": int(len(labels)),
        "n_detector_events": int(len(events)),
        "labels": labels,
        "detector_counts": detector_counts,
        "sorted_counts": sorted_counts[:, : detector_counts.shape[1]],
        "majority": float(max(np.bincount(labels, minlength=2)) / max(len(labels), 1)),
        "detector": FROZEN_REAL_DETECTOR_NAME,
    }


def _fit_eval(train_x, train_y, test_x, test_y) -> dict:
    if len(np.unique(train_y)) < 2 or len(test_y) == 0:
        return {"status": "skipped", "reason": "need both classes and a nonempty test set"}
    weights, bias = train(train_x, train_y.astype(float))
    pred = predict_labels(test_x, weights, bias)
    return classification(test_y, pred)


def run(
    primary_pid: str = DEFAULT_PID,
    secondary_pid: str = SECOND_PID,
    t0_s: float = 1465.0,
    duration_s: float = 40.0,
    secondary_t0_s: float = 1825.0,
) -> dict:
    primary = _session_block(primary_pid, t0_s, duration_s)
    try:
        secondary = _session_block(secondary_pid, secondary_t0_s, duration_s)
        secondary_error = None
    except Exception as exc:
        secondary = None
        secondary_error = f"{type(exc).__name__}: {exc}"

    seed = 20261103
    rng = np.random.default_rng(seed)
    labels = primary["labels"]
    train_ids, test_ids = [], []
    for label in (0, 1):
        ids = rng.permutation(np.flatnonzero(labels == label))
        split = max(1, int(0.7 * len(ids)))
        train_ids.extend(ids[:split])
        test_ids.extend(ids[split:])
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    within = {
        "sorted_counts": _fit_eval(primary["sorted_counts"][train_ids], labels[train_ids],
                                   primary["sorted_counts"][test_ids], labels[test_ids]),
        "detector_counts": _fit_eval(primary["detector_counts"][train_ids], labels[train_ids],
                                     primary["detector_counts"][test_ids], labels[test_ids]),
    }
    paired = None
    if "accuracy" in within["sorted_counts"] and "accuracy" in within["detector_counts"]:
        w, b = train(primary["sorted_counts"][train_ids], labels[train_ids].astype(float))
        pred_s = predict_labels(primary["sorted_counts"][test_ids], w, b)
        w2, b2 = train(primary["detector_counts"][train_ids], labels[train_ids].astype(float))
        pred_d = predict_labels(primary["detector_counts"][test_ids], w2, b2)
        paired = {
            "mcnemar_detector_vs_sorted": mcnemar_pvalue(labels[test_ids], pred_s, pred_d),
            "bootstrap_detector_minus_sorted": bootstrap_accuracy_delta(
                labels[test_ids], pred_s, pred_d, seed=seed
            ),
        }
    cross = {"status": "skipped", "error": secondary_error}
    if secondary is not None and secondary["n_trials"] >= 8 and primary["n_trials"] >= 8:
        n_features = min(primary["detector_counts"].shape[1], secondary["detector_counts"].shape[1])
        cross = {
            "status": "ran",
            "secondary_pid": secondary["pid"],
            "secondary_eid": secondary["eid"],
            "secondary_trials": secondary["n_trials"],
            "train_primary_test_secondary_detector": _fit_eval(
                primary["detector_counts"][:, :n_features], primary["labels"],
                secondary["detector_counts"][:, :n_features], secondary["labels"],
            ),
            "train_primary_test_secondary_sorted": _fit_eval(
                primary["sorted_counts"][:, :n_features], primary["labels"],
                secondary["sorted_counts"][:, :n_features], secondary["labels"],
            ),
            "secondary_majority": secondary["majority"],
        }
    return {
        "status": "ran",
        "frozen_detector": FROZEN_REAL_DETECTOR_NAME,
        "rationale": FROZEN_REAL_DETECTOR_RATIONALE,
        "rtl_follows": False,
        "rtl_reason": "Frozen path requires median CAR before thresholding; that stage is absent from the verified SystemVerilog pipeline.",
        "window": "50-250 ms after stimOn_times on a contiguous raw AP block",
        "primary": {k: v for k, v in primary.items() if k not in {"labels", "detector_counts", "sorted_counts"}},
        "within_session": within,
        "paired_tests": paired,
        "cross_recording": cross,
        "limitation": (
            "Detector counts are 32 selected contacts, not sorted units. "
            "Eligible visual trials are sparse, so the AP block is a 40 s trial-dense window "
            "(primary t0=1465 s, 11 trials; secondary t0=1825 s, 10 trials). "
            "Held-out accuracy on five primary trials is not a session-level result. "
            "Cross-recording uses unmatched contact indices across insertions. "
            "Sorting remains an imperfect comparison, not ground truth."
        ),
    }


def main() -> None:
    target = Path("reports/ibl-detector-decode.json")
    try:
        report = run()
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    target.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "within_session", "cross_recording", "error")}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
