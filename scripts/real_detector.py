"""Python detector path for real Neuropixels voltages.

The MEArec/RTL detector in ``scripts.detect_spikes`` stays unchanged: negative
six-sigma MAD thresholds at the recording sample rate, no common-average
reference, no artifact veto. This module is the research detector for native
30 kHz AP recordings. It is not bit-accurate RTL and does not replace the
hardware golden model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Sequence

import numpy as np
from scipy import signal

from scripts.detect_spikes import DetectorConfig, Event, detect_candidates, estimate_noise, match_events, merge_channel_events
from scripts.filter_iir import IirConfig, design_sos, maximum_pole_magnitude


@dataclass(frozen=True)
class RealDetectorConfig:
    """Offline real-recording detector. Defaults differ from the RTL 6σ path."""

    sample_rate_hz: float = 30_000.0
    highpass_hz: float = 300.0
    lowpass_hz: float | None = 6_000.0
    use_car: bool = True
    car_method: str = "median"
    threshold_sigma: float = 4.0
    polarity: str = "negative"
    artifact_channel_fraction: float = 0.5
    peak_search_ms: float = 0.5
    refractory_ms: float = 1.0
    merge_ms: float = 0.5
    match_tolerance_ms: float = 1.0
    mad_clip_sigma: float = 6.0
    mad_iterations: int = 3

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.threshold_sigma <= 0:
            raise ValueError("sample rate and threshold sigma must be positive")
        if self.car_method not in {"median", "mean"}:
            raise ValueError("car_method must be median or mean")
        if self.polarity not in {"negative", "positive", "both"}:
            raise ValueError("polarity must be negative, positive, or both")
        if not 0.0 < self.artifact_channel_fraction <= 1.0:
            raise ValueError("artifact_channel_fraction must be in (0, 1]")

    def detector_config(self) -> DetectorConfig:
        return DetectorConfig(
            sample_rate_hz=self.sample_rate_hz,
            threshold_sigma=self.threshold_sigma,
            peak_search_ms=self.peak_search_ms,
            refractory_ms=self.refractory_ms,
            merge_ms=self.merge_ms,
            match_tolerance_ms=self.match_tolerance_ms,
        )


def common_average_reference(samples: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract a per-sample common mode. Does not change the RTL filter."""
    traces = np.asarray(samples, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError("samples must be [samples, channels]")
    if method == "median":
        reference = np.median(traces, axis=1, keepdims=True)
    elif method == "mean":
        reference = np.mean(traces, axis=1, keepdims=True)
    else:
        raise ValueError("method must be median or mean")
    return traces - reference


def estimate_noise_robust(
    filtered_uv: np.ndarray,
    *,
    clip_sigma: float = 6.0,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative MAD that down-weights large transients after the first pass."""
    samples = np.asarray(filtered_uv, dtype=np.float64)
    baseline, sigma = estimate_noise(samples)
    if iterations <= 0:
        return baseline, sigma
    for _ in range(int(iterations)):
        next_baseline = np.empty_like(baseline)
        next_sigma = np.empty_like(sigma)
        for channel in range(samples.shape[1]):
            trace = samples[:, channel]
            keep = np.abs(trace - baseline[channel]) <= clip_sigma * sigma[channel]
            if int(np.count_nonzero(keep)) < 32:
                keep = np.ones(trace.shape[0], dtype=bool)
            subset = trace[keep][:, None]
            ch_base, ch_sigma = estimate_noise(subset)
            next_baseline[channel] = ch_base[0]
            next_sigma[channel] = ch_sigma[0]
        baseline, sigma = next_baseline, next_sigma
    return baseline, sigma


def _filter_voltages(voltages_uv: np.ndarray, config: RealDetectorConfig) -> tuple[np.ndarray, dict]:
    samples = np.asarray(voltages_uv, dtype=np.float64)
    nyquist = config.sample_rate_hz / 2.0
    if config.lowpass_hz is None:
        if not 0.0 < config.highpass_hz < nyquist:
            raise ValueError("high-pass must be below Nyquist")
        sos = signal.butter(2, config.highpass_hz, btype="highpass", fs=config.sample_rate_hz, output="sos")
    else:
        iir = IirConfig(
            sample_rate_hz=config.sample_rate_hz,
            highpass_hz=config.highpass_hz,
            lowpass_hz=config.lowpass_hz,
        )
        sos = design_sos(iir)
    filtered = signal.sosfilt(sos, samples, axis=0)
    info = {
        "sos": sos.tolist(),
        "maximum_pole_magnitude": maximum_pole_magnitude(sos),
        "highpass_hz": config.highpass_hz,
        "lowpass_hz": config.lowpass_hz,
    }
    return filtered, info


def preprocess(voltages_uv: np.ndarray, config: RealDetectorConfig) -> tuple[np.ndarray, dict]:
    filtered, info = _filter_voltages(voltages_uv, config)
    if config.use_car:
        filtered = common_average_reference(filtered, config.car_method)
    info["car"] = config.use_car
    info["car_method"] = config.car_method if config.use_car else None
    return filtered, info


def reject_spatial_artifacts(
    events: Sequence[Event],
    n_channels: int,
    config: RealDetectorConfig,
) -> tuple[list[Event], int]:
    """Drop simultaneous events that hit too many channels (likely artifacts)."""
    if not events:
        return [], 0
    merge_samples = max(1, config.detector_config().samples_for_ms(0.2))
    ordered = sorted(events, key=lambda event: event.sample)
    kept: list[Event] = []
    rejected = 0
    index = 0
    while index < len(ordered):
        start = ordered[index].sample
        group = [ordered[index]]
        index += 1
        while index < len(ordered) and ordered[index].sample - start <= merge_samples:
            group.append(ordered[index])
            index += 1
        channels = {event.channel for event in group}
        if len(channels) >= max(2, int(np.ceil(config.artifact_channel_fraction * n_channels))):
            rejected += len(group)
            continue
        kept.extend(group)
    return kept, rejected


def _detect_polarity(
    filtered_uv: np.ndarray,
    baseline: np.ndarray,
    sigma: np.ndarray,
    config: RealDetectorConfig,
    polarity: str,
) -> list[Event]:
    traces = filtered_uv if polarity == "negative" else -filtered_uv
    base = baseline if polarity == "negative" else -baseline
    events = detect_candidates(traces, base, sigma, config.detector_config())
    if polarity == "positive":
        return [
            Event(event.sample, event.channel, -event.voltage_uv, event.normalized_amplitude)
            for event in events
        ]
    return events


def detect_real(
    voltages_uv: np.ndarray,
    config: RealDetectorConfig | None = None,
) -> dict:
    """Filter, optionally CAR, detect, reject artifacts, and merge channels."""
    cfg = config or RealDetectorConfig()
    samples = np.asarray(voltages_uv, dtype=np.float64)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("voltages must be a finite [samples, channels] array")
    filtered, filter_info = preprocess(samples, cfg)
    baseline, sigma = estimate_noise_robust(
        filtered, clip_sigma=cfg.mad_clip_sigma, iterations=cfg.mad_iterations
    )
    polarities: Iterable[str]
    if cfg.polarity == "both":
        polarities = ("negative", "positive")
    else:
        polarities = (cfg.polarity,)
    candidates: list[Event] = []
    for polarity in polarities:
        candidates.extend(_detect_polarity(filtered, baseline, sigma, cfg, polarity))
    candidates = sorted(candidates, key=lambda event: (event.sample, event.channel))
    cleaned, rejected = reject_spatial_artifacts(candidates, samples.shape[1], cfg)
    merged = merge_channel_events(cleaned, cfg.detector_config())
    return {
        "config": asdict(cfg),
        "filter": {key: value for key, value in filter_info.items() if key != "sos"},
        "noise_sigma_uv": {
            "minimum": float(np.min(sigma)),
            "median": float(np.median(sigma)),
            "maximum": float(np.max(sigma)),
        },
        "baseline_uv": {
            "minimum": float(np.min(baseline)),
            "median": float(np.median(baseline)),
            "maximum": float(np.max(baseline)),
        },
        "channel_candidates": len(candidates),
        "artifact_rejected_candidates": rejected,
        "merged_detections": len(merged),
        "events": merged,
        "filtered_uv": filtered,
        "baseline": baseline,
        "noise_sigma": sigma,
    }


def sorter_aligned_snr(
    filtered_uv: np.ndarray,
    baseline: np.ndarray,
    sigma: np.ndarray,
    truth_seconds: np.ndarray,
    sample_rate_hz: float,
) -> dict:
    """Measure trough depth at imperfect sorter times; not ground-truth SNR."""
    times = np.asarray(truth_seconds, dtype=float)
    if times.size == 0:
        return {"n": 0, "median_peak_sigma": None, "fraction_above_4_sigma": None, "fraction_above_6_sigma": None}
    samples = np.rint(times * sample_rate_hz).astype(int)
    samples = samples[(samples >= 0) & (samples < filtered_uv.shape[0])]
    peaks = []
    window = max(1, int(round(0.5e-3 * sample_rate_hz)))
    for sample in samples:
        start = max(0, sample - window)
        stop = min(filtered_uv.shape[0], sample + window + 1)
        local = filtered_uv[start:stop]
        trough = np.min((local - baseline) / sigma, axis=1)
        peaks.append(float(-np.min(trough)))
    peaks_arr = np.asarray(peaks, dtype=float)
    return {
        "n": int(len(peaks_arr)),
        "median_peak_sigma": float(np.median(peaks_arr)),
        "p90_peak_sigma": float(np.percentile(peaks_arr, 90)),
        "fraction_above_4_sigma": float(np.mean(peaks_arr >= 4.0)),
        "fraction_above_6_sigma": float(np.mean(peaks_arr >= 6.0)),
    }


def evaluate_against_reference(
    detections: Sequence[Event],
    reference_seconds: np.ndarray,
    config: RealDetectorConfig,
    tolerances_ms: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict:
    """Match detections to an imperfect spike-sorting reference at several windows."""
    out = {}
    for tolerance in tolerances_ms:
        detector = replace(config.detector_config(), match_tolerance_ms=float(tolerance))
        _, metrics = match_events(list(detections), np.asarray(reference_seconds, dtype=float), detector)
        out[str(tolerance)] = metrics
    return out


LEGACY_SIX_SIGMA = RealDetectorConfig(
    sample_rate_hz=30_000.0,
    use_car=False,
    threshold_sigma=6.0,
    polarity="negative",
    artifact_channel_fraction=1.0,
    mad_iterations=0,
)
