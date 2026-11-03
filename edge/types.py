"""Shared dataclasses for the ASIC-to-edge stack. Keep this module free of algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge import N_CHANNELS, SAMPLE_RATE_HZ, SNIPPET_POST_MS, SNIPPET_PRE_MS


@dataclass(frozen=True)
class ElectrodeSite:
    channel_id: int
    xyz_um: tuple[float, float, float]
    label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "xyz_um": list(self.xyz_um),
            "label": self.label,
        }


@dataclass(frozen=True)
class ElectrodeArray:
    sites: tuple[ElectrodeSite, ...]
    description: str = ""
    frame: str = "probe_um"

    @property
    def n_channels(self) -> int:
        return len(self.sites)

    def channel_ids(self) -> tuple[int, ...]:
        return tuple(site.channel_id for site in self.sites)

    def xyz_um(self) -> np.ndarray:
        return np.asarray([site.xyz_um for site in self.sites], dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "n_channels": self.n_channels,
            "frame": self.frame,
            "sites": [site.to_dict() for site in self.sites],
        }


@dataclass(frozen=True)
class Snippet:
    """Multichannel waveform window aligned to a detected event."""

    event_id: int
    t_sample: int
    t0_sample: int
    sample_rate_hz: float
    channel_ids: tuple[int, ...]
    voltages_uv: np.ndarray
    electrode_xyz_um: np.ndarray
    peak_channel: int | None = None
    ground_truth_xyz_um: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.voltages_uv)
        if samples.ndim != 2:
            raise ValueError("voltages_uv must be [samples, channels]")
        if samples.shape[1] != len(self.channel_ids):
            raise ValueError("channel count does not match voltages_uv")
        if self.electrode_xyz_um.shape != (len(self.channel_ids), 3):
            raise ValueError("electrode_xyz_um must be [channels, 3]")
        if not np.isfinite(samples).all():
            raise ValueError("snippet voltages must be finite")

    @property
    def n_pre(self) -> int:
        return int(self.t_sample - self.t0_sample)


@dataclass(frozen=True)
class LocalizationEstimate:
    available: bool
    method: str
    xyz_um: tuple[float, float, float] | None
    ground_truth_xyz_um: tuple[float, float, float] | None
    error_um: float | None
    latency_s: float
    confidence: float | None
    reason: str
    event_id: int | None = None
    unverified: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "method": self.method,
            "xyz_um": None if self.xyz_um is None else list(self.xyz_um),
            "ground_truth_xyz_um": None
            if self.ground_truth_xyz_um is None
            else list(self.ground_truth_xyz_um),
            "error_um": self.error_um,
            "latency_s": self.latency_s,
            "confidence": self.confidence,
            "reason": self.reason,
            "event_id": self.event_id,
            "unverified": self.unverified,
        }

    def to_session_dict(self, xyz_mm: list[float] | None = None) -> dict[str, object]:
        return {
            "available": self.available,
            "method": self.method,
            "xyz_mm": xyz_mm,
            "error_um": self.error_um,
            "confidence": self.confidence,
            "unverified": self.unverified,
            "reason": self.reason,
        }


def snippet_sample_counts(sample_rate_hz: float = SAMPLE_RATE_HZ) -> tuple[int, int]:
    n_pre = int(round(SNIPPET_PRE_MS * sample_rate_hz / 1000.0))
    n_post = int(round(SNIPPET_POST_MS * sample_rate_hz / 1000.0))
    return n_pre, n_post


def empty_snippet(n_channels: int = N_CHANNELS) -> Snippet:
    n_pre, n_post = snippet_sample_counts()
    n_samples = n_pre + n_post
    zeros = np.zeros((n_samples, n_channels), dtype=np.float64)
    xyz = np.zeros((n_channels, 3), dtype=np.float64)
    return Snippet(
        event_id=0,
        t_sample=n_pre,
        t0_sample=0,
        sample_rate_hz=SAMPLE_RATE_HZ,
        channel_ids=tuple(range(n_channels)),
        voltages_uv=zeros,
        electrode_xyz_um=xyz,
    )


@dataclass
class SessionLocalization:
    """JSON contract consumed by the visualization session snapshot."""

    available: bool = False
    method: str = "none"
    xyz_mm: tuple[float, float, float] | None = None
    error_um: float | None = None
    confidence: float | None = None
    unverified: bool = True
    reason: str = "Localization not implemented"

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "method": self.method,
            "xyz_mm": None if self.xyz_mm is None else list(self.xyz_mm),
            "error_um": self.error_um,
            "confidence": self.confidence,
            "unverified": self.unverified,
            "reason": self.reason,
        }
