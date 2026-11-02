"""Load and replay sorted Neuropixels spike events. Not raw voltage."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from viz.sim.events import NeuralEvent
from viz.sim.regions import (
    NEUROPIXELS_ALLOWED_REGIONS,
    get_region,
    lerp_in_region,
    map_mouse_area,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "neuropixels_visual_coding_mini.json"
DISCLAIMER = (
    "Anatomical human brain surface with illustrative wiring. Mouse visual-cortex "
    "Neuropixels shown on occipital/visual cortex only. No measured axonal paths."
)


@dataclass(frozen=True)
class ProbeChannel:
    channel_id: int
    local_xyz_um: tuple[float, float, float]
    region_acronym: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "local_xyz_um": list(self.local_xyz_um),
            "region_acronym": self.region_acronym,
        }


@dataclass
class SortedUnit:
    unit_id: str
    peak_channel: int
    region_acronym: str
    spike_times_s: np.ndarray
    amplitudes: np.ndarray
    ccf_ap_dv_lr_mm: tuple[float, float, float] | None
    display_xyz_mm: tuple[float, float, float]
    position_quality: str


@dataclass
class NeuropixelsRecording:
    dataset: str
    species: str
    session_id: str
    disclaimer: str
    stimulus: list[dict[str, Any]]
    channels: list[ProbeChannel]
    units: list[SortedUnit]
    duration_s: float
    source_path: str
    has_raw_voltage: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "species": self.species,
            "session_id": self.session_id,
            "disclaimer": self.disclaimer,
            "n_units": len(self.units),
            "n_channels": len(self.channels),
            "duration_s": self.duration_s,
            "has_raw_voltage": self.has_raw_voltage,
            "source_path": self.source_path,
            "playback_kind": "sorted_spikes",
        }


def _stable_display_xyz(
    unit_id: str,
    region_id: str,
    ccf: tuple[float, float, float] | None,
    channel: ProbeChannel | None,
    channel_index: int,
    n_channels: int,
) -> tuple[tuple[float, float, float], str]:
    region = get_region(region_id)
    if region_id not in NEUROPIXELS_ALLOWED_REGIONS:
        raise ValueError("Neuropixels events cannot leave the supported visual-cortex volume")
    if ccf is not None:
        # Preserve relative CCF offsets inside the stylized occipital ellipsoid.
        ap, dv, lr = ccf
        u = 0.35 + 0.3 * (1.0 / (1.0 + np.exp(-lr)))
        v = 0.35 + 0.3 * (1.0 / (1.0 + np.exp(-ap)))
        w = 0.2 + 0.4 * (1.0 / (1.0 + np.exp(-dv)))
        return lerp_in_region(region, float(u), float(v), float(w)), "unit_ccf"
    if channel is not None and n_channels > 0:
        frac = channel_index / max(n_channels - 1, 1)
        return lerp_in_region(region, 0.5, frac, 0.45), "electrode_channel"
    digest = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16)
    u = (digest % 100) / 100.0
    v = ((digest // 100) % 100) / 100.0
    w = 0.4
    return lerp_in_region(region, u, v, w), "unknown"


def load_recording(path: Path | None = None) -> NeuropixelsRecording:
    source = Path(path) if path is not None else FIXTURE_PATH
    if source.suffix.lower() == ".nwb":
        return _load_nwb(source)
    payload = json.loads(source.read_text())
    return _from_payload(payload, source)


def _from_payload(payload: dict[str, Any], source: Path) -> NeuropixelsRecording:
    if payload.get("has_raw_voltage"):
        raise ValueError("this loader plays sorted spikes only; refuse raw-voltage pretence")
    channels = [
        ProbeChannel(
            channel_id=int(row["channel_id"]),
            local_xyz_um=(
                float(row["x_um"]),
                float(row["y_um"]),
                float(row["z_um"]),
            ),
            region_acronym=str(row["region"]),
        )
        for row in payload["channels"]
    ]
    channel_by_id = {channel.channel_id: channel for channel in channels}
    channel_index = {channel.channel_id: index for index, channel in enumerate(channels)}
    units: list[SortedUnit] = []
    duration = 0.0
    for row in payload["units"]:
        acronym = str(row["region"])
        region_id = map_mouse_area(acronym)
        times = np.asarray(row["spike_times_s"], dtype=float)
        amplitudes = np.asarray(row.get("amplitudes", np.full(times.shape, -80.0)), dtype=float)
        if times.size and amplitudes.size != times.size:
            raise ValueError(f"unit {row['unit_id']} amplitude length mismatch")
        ccf = None
        if row.get("ccf_ap_mm") is not None:
            ccf = (
                float(row["ccf_ap_mm"]),
                float(row["ccf_dv_mm"]),
                float(row["ccf_lr_mm"]),
            )
        peak_channel = int(row["peak_channel"])
        display_xyz, quality = _stable_display_xyz(
            str(row["unit_id"]),
            region_id,
            ccf,
            channel_by_id.get(peak_channel),
            channel_index.get(peak_channel, 0),
            len(channels),
        )
        if not get_region(region_id).contains(display_xyz, margin=1.05):
            raise ValueError("mapped Neuropixels coordinate escaped the visual-cortex volume")
        units.append(
            SortedUnit(
                unit_id=str(row["unit_id"]),
                peak_channel=peak_channel,
                region_acronym=acronym,
                spike_times_s=times,
                amplitudes=amplitudes,
                ccf_ap_dv_lr_mm=ccf,
                display_xyz_mm=display_xyz,
                position_quality=quality,
            )
        )
        if times.size:
            duration = max(duration, float(times.max()))
    return NeuropixelsRecording(
        dataset=str(payload.get("dataset", "unknown")),
        species=str(payload.get("species", "Mus musculus")),
        session_id=str(payload.get("session_id", "unknown")),
        disclaimer=str(payload.get("disclaimer", DISCLAIMER)),
        stimulus=list(payload.get("stimulus", [])),
        channels=channels,
        units=units,
        duration_s=max(duration, float(payload.get("duration_s", 0.0))),
        source_path=str(source),
        has_raw_voltage=False,
    )


def _load_nwb(path: Path) -> NeuropixelsRecording:
    try:
        import pynwb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pynwb is not installed. Use the JSON fixture or install optional Neuropixels extras. "
            "Sorted-spike playback does not validate the ADC pipeline."
        ) from exc
    raise RuntimeError(
        f"NWB import is available ({pynwb.__name__}) but this repository ships a JSON fixture "
        f"instead of a full Allen session. Point the loader at {FIXTURE_PATH.name} or convert "
        f"{path} to the fixture schema. Raw voltage, if present in NWB, is not played as spikes."
    )


@dataclass
class NeuropixelsPlayer:
    recording: NeuropixelsRecording
    loop: bool = True
    cursor: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.cursor = {unit.unit_id: 0 for unit in self.recording.units}

    def generate(self, t_start_s: float, t_end_s: float) -> list[NeuralEvent]:
        duration = max(self.recording.duration_s, 1e-6)
        events: list[NeuralEvent] = []
        start_cycle = int(np.floor(max(t_start_s, 0.0) / duration))
        end_cycle = int(np.floor(max(t_end_s - 1e-12, 0.0) / duration))
        if not self.loop:
            start_cycle = 0
            end_cycle = 0
        for unit in self.recording.units:
            times = unit.spike_times_s
            if times.size == 0:
                continue
            cycles = range(start_cycle, end_cycle + 1) if self.loop else [0]
            for cycle in cycles:
                for index, original_t in enumerate(times.tolist()):
                    session_t = float(original_t) + (cycle * duration if self.loop else 0.0)
                    if session_t < t_start_s or session_t >= t_end_s:
                        continue
                    amplitude = float(unit.amplitudes[index % len(unit.amplitudes)])
                    region_id = map_mouse_area(unit.region_acronym)
                    events.append(
                        NeuralEvent(
                            event_id=f"np:{unit.unit_id}:{cycle}:{index}",
                            t_s=session_t,
                            xyz_mm=unit.display_xyz_mm,
                            region_id=region_id,
                            amplitude=amplitude,
                            source_id=unit.unit_id,
                            kind="neuropixels_spike",
                            duration_ms=1.0,
                            coordinate_frame="stylized_human",
                            position_quality=unit.position_quality,
                            amplitude_units="uv",
                            channel_id=unit.peak_channel,
                            metadata={
                                "dataset": self.recording.dataset,
                                "species": self.recording.species,
                                "original_t_s": float(original_t),
                                "allen_ccf_v3_mm": unit.ccf_ap_dv_lr_mm,
                                "mouse_area": unit.region_acronym,
                                "playback": "sorted_spikes",
                                "not_raw_voltage": True,
                            },
                        )
                    )
        events.sort(key=lambda event: (event.t_s, event.event_id))
        return events
