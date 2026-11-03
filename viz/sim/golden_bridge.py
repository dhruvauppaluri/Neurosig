"""Run the existing float golden model on simulated voltages. Do not alter DSP code."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.detect_spikes import (  # noqa: E402
    DetectorConfig,
    Event,
    detect_candidates,
    estimate_noise,
    match_events,
    merge_channel_events,
)
from scripts.filter_iir import IirConfig, design_sos, filter_adc_codes  # noqa: E402
from scripts.quantize_adc import AdcConfig, quantize_voltages  # noqa: E402

from edge.types import snippet_sample_counts  # noqa: E402
from viz.sim.events import NeuralEvent
from viz.sim.localization import (
    EdgeLocalizerAdapter,
    ElectrodeArray,
    LocalizationRequest,
    NotImplementedLocalizer,
    empty_session_localization,
)
from viz.sim.recording_model import RecordingConfig, render_voltages


@dataclass(frozen=True)
class GoldenRun:
    voltages_uv: np.ndarray
    codes: np.ndarray
    filtered_uv: np.ndarray
    detections: list[Event]
    merged: list[Event]
    metrics: dict[str, Any]
    localization: dict[str, Any]
    events: list[NeuralEvent]
    snippets: dict[str, dict[str, Any]]


def _site_xyz(array: ElectrodeArray, channel: int) -> tuple[float, float, float]:
    return array.sites[channel].xyz_mm


def _snippet_payload(
    event_id: str,
    voltages: np.ndarray,
    channel_ids: tuple[int, ...],
    t0_sample: int,
    t_sample: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "sample_rate_hz": float(sample_rate_hz),
        "channel_ids": [int(channel) for channel in channel_ids],
        "t0_sample": int(t0_sample),
        "t_sample": int(t_sample),
        "voltages_uv": np.asarray(voltages, dtype=np.float64).tolist(),
    }


def _extract_window(voltages: np.ndarray, t_sample: int, n_pre: int, n_post: int) -> tuple[np.ndarray, int]:
    n_samples = int(voltages.shape[0])
    t0 = max(0, int(t_sample) - int(n_pre))
    t1 = min(n_samples, int(t_sample) + int(n_post))
    if t1 <= t0:
        return np.zeros((0, voltages.shape[1]), dtype=np.float64), t0
    return np.asarray(voltages[t0:t1], dtype=np.float64), t0


def _matching_source(
    detected: Event,
    source_events: list[NeuralEvent],
    window_start_s: float,
    detector: DetectorConfig,
) -> NeuralEvent | None:
    tolerance = detector.samples_for_ms(detector.match_tolerance_ms)
    best: NeuralEvent | None = None
    best_delta = tolerance + 1
    for event in source_events:
        if event.kind != "synthetic_source":
            continue
        sample = int(round((event.t_s - window_start_s) * detector.sample_rate_hz))
        delta = abs(sample - detected.sample)
        if delta <= tolerance and delta < best_delta:
            best = event
            best_delta = delta
    return best


def process_window(
    source_events: list[NeuralEvent],
    window_start_s: float,
    window_duration_s: float,
    *,
    recording: RecordingConfig | None = None,
    adc: AdcConfig | None = None,
    detector: DetectorConfig | None = None,
    run_detector: bool = True,
) -> GoldenRun:
    voltages, array = render_voltages(source_events, window_start_s, window_duration_s, config=recording)
    adc_config = adc or AdcConfig()
    codes, _saturated = quantize_voltages(voltages, adc_config)
    sos = design_sos(IirConfig())
    filtered = filter_adc_codes(codes, adc_config.lsb_uv, sos)
    detector_config = detector or DetectorConfig()
    localization = empty_session_localization()
    snippets: dict[str, dict[str, Any]] = {}

    emitted: list[NeuralEvent] = []
    detections: list[Event] = []
    merged: list[Event] = []
    metrics: dict[str, Any] = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ground_truth_available": bool(source_events),
        "note": "offline golden-model window; not RTL",
    }

    for event in source_events:
        if event.kind != "synthetic_source":
            continue
        nearest = min(
            array.sites,
            key=lambda site: float(np.linalg.norm(np.array(site.xyz_mm) - np.array(event.xyz_mm))),
        )
        emitted.append(
            NeuralEvent(
                event_id=f"el:{event.event_id}",
                t_s=event.t_s,
                xyz_mm=nearest.xyz_mm,
                region_id="visual_cortex",
                amplitude=event.amplitude,
                source_id=event.source_id,
                kind="electrode_activity",
                duration_ms=event.duration_ms,
                position_quality="electrode_channel",
                channel_id=nearest.channel_id,
                ground_truth_event_id=event.event_id,
                metadata={"model": "point_source_attenuation"},
            )
        )

    if run_detector and voltages.shape[0] >= 32:
        baseline, sigma = estimate_noise(filtered)
        detections = detect_candidates(filtered, baseline, sigma, detector_config)
        merged = merge_channel_events(detections, detector_config)
        truth_seconds = np.array(
            [event.t_s - window_start_s for event in source_events if event.kind == "synthetic_source"],
            dtype=float,
        )
        _matches, metrics = match_events(merged, truth_seconds, detector_config)
        metrics["ground_truth_available"] = bool(source_events)
        metrics["note"] = "offline golden-model window; not RTL"
        matched_truth = {int(round(t * detector_config.sample_rate_hz)) for t in truth_seconds}
        used_truth: set[int] = set()
        channel_ids = tuple(site.channel_id for site in array.sites)
        electrode_xyz = tuple(site.xyz_mm for site in array.sites)
        n_pre, n_post = snippet_sample_counts(detector_config.sample_rate_hz)
        adapter = EdgeLocalizerAdapter()
        last_estimate = localization
        preferred_estimate = None
        for detected in merged:
            abs_t = window_start_s + detected.sample / detector_config.sample_rate_hz
            det_id = f"det:{detected.sample}:{detected.channel}"
            emitted.append(
                NeuralEvent(
                    event_id=det_id,
                    t_s=abs_t,
                    xyz_mm=_site_xyz(array, detected.channel),
                    region_id="visual_cortex",
                    amplitude=detected.voltage_uv,
                    source_id="golden_float",
                    kind="detected_spike",
                    duration_ms=1.0,
                    position_quality="electrode_channel",
                    amplitude_units="uv",
                    channel_id=detected.channel,
                    metadata={"normalized_amplitude": detected.normalized_amplitude, "rtl": False},
                )
            )
            window, t0_sample = _extract_window(voltages, detected.sample, n_pre, n_post)
            if window.size:
                payload = _snippet_payload(
                    det_id, window, channel_ids, t0_sample, detected.sample, detector_config.sample_rate_hz
                )
                snippets[det_id] = payload
                matched = _matching_source(detected, source_events, window_start_s, detector_config)
                estimate = adapter.estimate(
                    LocalizationRequest(
                        channel_ids=channel_ids,
                        electrode_xyz_mm=electrode_xyz,
                        event_ids=(det_id,),
                        snippets_uv=window,
                        sample_rate_hz=detector_config.sample_rate_hz,
                        t_sample=detected.sample,
                        t0_sample=t0_sample,
                        peak_channel=detected.channel,
                        ground_truth_xyz_mm=None if matched is None else matched.xyz_mm,
                    )
                )
                last_estimate = estimate.to_dict()
                if last_estimate.get("available") and last_estimate.get("error_um") is not None:
                    preferred_estimate = last_estimate
                if estimate.available and estimate.xyz_mm is not None:
                    loc_id = f"loc:{detected.sample}:{detected.channel}"
                    snippets[loc_id] = {**payload, "event_id": loc_id}
                    emitted.append(
                        NeuralEvent(
                            event_id=loc_id,
                            t_s=abs_t,
                            xyz_mm=estimate.xyz_mm,
                            region_id="visual_cortex",
                            amplitude=detected.voltage_uv,
                            source_id="edge_localizer",
                            kind="localization_estimate",
                            duration_ms=1.0,
                            position_quality="simulated" if matched is not None else "unknown",
                            amplitude_units="uv",
                            channel_id=detected.channel,
                            ground_truth_event_id=None if matched is None else matched.event_id,
                            metadata={
                                "method": estimate.method,
                                "error_um": estimate.error_um,
                                "confidence": estimate.confidence,
                                "unverified": True,
                                "reason": estimate.reason,
                            },
                        )
                    )
        if preferred_estimate:
            localization = preferred_estimate
        elif last_estimate:
            localization = last_estimate
        probe_xyz = np.array([site.xyz_mm for site in array.sites], dtype=float)
        for event in source_events:
            if event.kind != "synthetic_source":
                continue
            sample = int(round((event.t_s - window_start_s) * detector_config.sample_rate_hz))
            close = any(
                abs(sample - detected.sample)
                <= detector_config.samples_for_ms(detector_config.match_tolerance_ms)
                for detected in merged
            )
            near_probe = float(np.min(np.linalg.norm(probe_xyz - np.array(event.xyz_mm), axis=1))) < 4.0
            if not close and near_probe:
                emitted.append(
                    NeuralEvent(
                        event_id=f"miss:{event.event_id}",
                        t_s=event.t_s,
                        xyz_mm=event.xyz_mm,
                        region_id=event.region_id,
                        amplitude=event.amplitude,
                        source_id=event.source_id,
                        kind="missed_spike",
                        duration_ms=event.duration_ms,
                        position_quality="simulated",
                        ground_truth_event_id=event.event_id,
                    )
                )
            elif close:
                used_truth.add(sample)
        del matched_truth, used_truth
    else:
        localization = NotImplementedLocalizer().estimate(
            LocalizationRequest(
                channel_ids=tuple(site.channel_id for site in array.sites),
                electrode_xyz_mm=tuple(site.xyz_mm for site in array.sites),
                event_ids=tuple(event.event_id for event in source_events),
            )
        ).to_dict()

    return GoldenRun(
        voltages_uv=voltages,
        codes=codes,
        filtered_uv=filtered,
        detections=detections,
        merged=merged,
        metrics=metrics,
        localization=localization,
        events=emitted,
        snippets=snippets,
    )


def refuse_spike_timestamps_as_voltage(kind: str) -> None:
    if kind == "neuropixels_spike":
        raise ValueError(
            "Cannot feed sorted Neuropixels spike timestamps into the ADC pipeline. "
            "Mode B is event playback only unless raw voltage recordings are provided."
        )
