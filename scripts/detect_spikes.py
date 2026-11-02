"""Detect negative extracellular spikes in the floating-point IIR output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import h5py
import matplotlib.pyplot as plt
import numpy as np


MAD_NORMAL_SCALE = 0.6744897501960817


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate_hz: float = 32_000.0
    threshold_sigma: float = 6.0
    peak_search_ms: float = 0.5
    refractory_ms: float = 1.0
    merge_ms: float = 0.5
    match_tolerance_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.threshold_sigma <= 0:
            raise ValueError("sample rate and threshold sigma must be positive")
        if min(self.peak_search_ms, self.refractory_ms, self.merge_ms, self.match_tolerance_ms) < 0:
            raise ValueError("time windows cannot be negative")

    def samples_for_ms(self, duration_ms: float) -> int:
        return int(round(duration_ms * self.sample_rate_hz / 1000.0))


@dataclass(frozen=True)
class Event:
    sample: int
    channel: int
    voltage_uv: float
    normalized_amplitude: float


def estimate_noise(filtered_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate channel baselines and noise sigma using the spike-resistant MAD."""
    samples = np.asarray(filtered_uv, dtype=np.float64)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("filtered data must be a finite [samples, channels] array")
    baseline = np.median(samples, axis=0)
    sigma = np.median(np.abs(samples - baseline), axis=0) / MAD_NORMAL_SCALE
    if np.any(sigma <= 0):
        raise ValueError("every channel must have a positive noise estimate")
    return baseline, sigma


def detect_candidates(
    filtered_uv: np.ndarray,
    baseline_uv: np.ndarray,
    noise_sigma_uv: np.ndarray,
    config: DetectorConfig,
) -> list[Event]:
    """Detect channel-local threshold crossings and timestamp their local minima."""
    samples = np.asarray(filtered_uv, dtype=np.float64)
    search = config.samples_for_ms(config.peak_search_ms)
    refractory = config.samples_for_ms(config.refractory_ms)
    events: list[Event] = []

    for channel in range(samples.shape[1]):
        trace = samples[:, channel]
        threshold = baseline_uv[channel] - config.threshold_sigma * noise_sigma_uv[channel]
        crossings = np.flatnonzero((trace[:-1] >= threshold) & (trace[1:] < threshold)) + 1
        last_peak = -refractory - 1
        for crossing in crossings:
            if crossing - last_peak < refractory:
                continue
            stop = min(len(trace), crossing + search + 1)
            peak = int(crossing + np.argmin(trace[crossing:stop]))
            if peak - last_peak < refractory:
                continue
            normalized = float((baseline_uv[channel] - trace[peak]) / noise_sigma_uv[channel])
            events.append(Event(peak, channel, float(trace[peak]), normalized))
            last_peak = peak
    return sorted(events, key=lambda event: event.sample)


def merge_channel_events(events: list[Event], config: DetectorConfig) -> list[Event]:
    """Merge one neural event seen on several electrodes, retaining its strongest channel."""
    if not events:
        return []
    merge_samples = config.samples_for_ms(config.merge_ms)
    groups: list[list[Event]] = []
    for event in sorted(events, key=lambda candidate: candidate.sample):
        if not groups or event.sample - groups[-1][0].sample > merge_samples:
            groups.append([event])
        else:
            groups[-1].append(event)
    return [max(group, key=lambda candidate: candidate.normalized_amplitude) for group in groups]


def load_ground_truth(path: Path) -> np.ndarray:
    """Load and combine MEArec spike times, in seconds, from every simulated unit."""
    with h5py.File(path, "r") as recording:
        if "spiketrains" not in recording:
            raise KeyError(f"{path} has no spiketrains group")
        trains = [np.asarray(recording[f"spiketrains/{key}/times"][()]) for key in recording["spiketrains"]]
    return np.sort(np.concatenate(trains)) if trains else np.empty(0, dtype=float)


def match_events(
    detections: list[Event], ground_truth_seconds: np.ndarray, config: DetectorConfig
) -> tuple[list[dict], dict]:
    """Perform an ordered one-to-one match within the configured timing tolerance."""
    detected_samples = np.array([event.sample for event in detections], dtype=int)
    truth_samples = np.rint(np.asarray(ground_truth_seconds) * config.sample_rate_hz).astype(int)
    tolerance = config.samples_for_ms(config.match_tolerance_ms)
    matches: list[dict] = []
    detection_index = truth_index = 0
    while detection_index < len(detected_samples) and truth_index < len(truth_samples):
        difference = int(detected_samples[detection_index] - truth_samples[truth_index])
        if abs(difference) <= tolerance:
            matches.append(
                {
                    "detection_index": detection_index,
                    "truth_index": truth_index,
                    "timing_error_ms": difference / config.sample_rate_hz * 1000.0,
                }
            )
            detection_index += 1
            truth_index += 1
        elif difference < -tolerance:
            detection_index += 1
        else:
            truth_index += 1

    true_positives = len(matches)
    false_positives = len(detections) - true_positives
    false_negatives = len(truth_samples) - true_positives
    precision = true_positives / len(detections) if detections else 0.0
    recall = true_positives / len(truth_samples) if len(truth_samples) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.array([match["timing_error_ms"] for match in matches], dtype=float)
    metrics = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_timing_error_ms": float(np.mean(np.abs(errors))) if len(errors) else None,
        "maximum_absolute_timing_error_ms": float(np.max(np.abs(errors))) if len(errors) else None,
    }
    return matches, metrics


def save_detection_plot(
    path: Path,
    filtered_uv: np.ndarray,
    baseline_uv: np.ndarray,
    noise_sigma_uv: np.ndarray,
    detections: list[Event],
    truth_seconds: np.ndarray,
    config: DetectorConfig,
) -> None:
    normalized = (filtered_uv - baseline_uv) / noise_sigma_uv
    strongest_negative = np.min(normalized, axis=1)
    times = np.arange(len(strongest_negative)) / config.sample_rate_hz
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, strongest_negative, linewidth=0.5, label="most negative channel")
    ax.axhline(-config.threshold_sigma, color="tab:orange", linestyle="--", label="threshold")
    for index, event in enumerate(detections):
        ax.axvline(event.sample / config.sample_rate_hz, color="tab:red", alpha=0.55,
                   label="detected" if index == 0 else None)
    for index, time_s in enumerate(truth_seconds):
        ax.axvline(time_s, color="tab:green", linestyle=":", alpha=0.8,
                   label="ground truth" if index == 0 else None)
    ax.set(xlabel="Time (s)", ylabel="Normalized voltage (σ)",
           title="Floating-point spike detections versus MEArec ground truth")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=Path("data/filtered/filtered_uv.npy"))
    parser.add_argument("--recording", type=Path, default=Path("data/unfiltered/recording.h5"))
    parser.add_argument("--output", type=Path, default=Path("data/detections"))
    parser.add_argument("--sample-rate", type=float, default=32_000.0)
    parser.add_argument("--threshold-sigma", type=float, default=6.0)
    parser.add_argument("--peak-search-ms", type=float, default=0.5)
    parser.add_argument("--refractory-ms", type=float, default=1.0)
    parser.add_argument("--merge-ms", type=float, default=0.5)
    parser.add_argument("--match-tolerance-ms", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectorConfig(
        args.sample_rate,
        args.threshold_sigma,
        args.peak_search_ms,
        args.refractory_ms,
        args.merge_ms,
        args.match_tolerance_ms,
    )
    filtered_uv = np.load(args.filtered)
    baseline_uv, noise_sigma_uv = estimate_noise(filtered_uv)
    candidates = detect_candidates(filtered_uv, baseline_uv, noise_sigma_uv, config)
    detections = merge_channel_events(candidates, config)
    truth_seconds = load_ground_truth(args.recording)
    matches, metrics = match_events(detections, truth_seconds, config)

    event_rows = [
        {
            "sample": event.sample,
            "time_s": event.sample / config.sample_rate_hz,
            "channel": event.channel,
            "voltage_uv": event.voltage_uv,
            "normalized_amplitude_sigma": event.normalized_amplitude,
        }
        for event in detections
    ]
    report = {
        "filtered_input": str(args.filtered),
        "ground_truth_input": str(args.recording),
        "shape": list(filtered_uv.shape),
        "detector": {
            **asdict(config),
            "threshold_polarity": "negative",
            "noise_estimator": "median absolute deviation / 0.6744897501960817, per channel",
            "timestamp_rule": "minimum following the negative threshold crossing",
            "cross_channel_merge_rule": "within merge window, retain greatest normalized negative amplitude",
        },
        "noise_sigma_uv": {
            "minimum": float(np.min(noise_sigma_uv)),
            "median": float(np.median(noise_sigma_uv)),
            "maximum": float(np.max(noise_sigma_uv)),
        },
        "channel_candidates": len(candidates),
        "merged_detections": len(detections),
        "ground_truth_spikes": len(truth_seconds),
        "events": event_rows,
        "matches": matches,
        "metrics": metrics,
        "limitation": "Metrics describe one seeded one-second recording with one neuron; validate on longer multi-unit recordings.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "events.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=event_rows[0].keys() if event_rows else ["sample", "time_s", "channel", "voltage_uv", "normalized_amplitude_sigma"])
        writer.writeheader()
        writer.writerows(event_rows)
    save_detection_plot(
        args.output / "detections.png",
        filtered_uv,
        baseline_uv,
        noise_sigma_uv,
        detections,
        truth_seconds,
        config,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote floating-point detections to {args.output.resolve()}")


if __name__ == "__main__":
    main()
