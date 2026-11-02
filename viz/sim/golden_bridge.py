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

from viz.sim.events import NeuralEvent
from viz.sim.localization import ElectrodeArray, LocalizationRequest, NotImplementedLocalizer
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


def _site_xyz(array: ElectrodeArray, channel: int) -> tuple[float, float, float]:
    return array.sites[channel].xyz_mm


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
    localizer = NotImplementedLocalizer()
    localization = localizer.estimate(
        LocalizationRequest(
            channel_ids=tuple(site.channel_id for site in array.sites),
            electrode_xyz_mm=tuple(site.xyz_mm for site in array.sites),
            event_ids=tuple(event.event_id for event in source_events),
        )
    ).to_dict()

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
        for detected in merged:
            abs_t = window_start_s + detected.sample / detector_config.sample_rate_hz
            emitted.append(
                NeuralEvent(
                    event_id=f"det:{detected.sample}:{detected.channel}",
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

    return GoldenRun(
        voltages_uv=voltages,
        codes=codes,
        filtered_uv=filtered,
        detections=detections,
        merged=merged,
        metrics=metrics,
        localization=localization,
        events=emitted,
    )


def refuse_spike_timestamps_as_voltage(kind: str) -> None:
    if kind == "neuropixels_spike":
        raise ValueError(
            "Cannot feed sorted Neuropixels spike timestamps into the ADC pipeline. "
            "Mode B is event playback only unless raw voltage recordings are provided."
        )
