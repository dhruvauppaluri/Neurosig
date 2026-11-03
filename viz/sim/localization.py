"""Visualization localization adapter. Algorithms live in edge.localize, not here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from edge.coords import display_mm_to_probe_um, euclidean_um, in_probe_volume_um, probe_um_to_display_mm
from edge.types import LocalizationEstimate, Snippet

SESSION_LOCALIZATION_KEYS = (
    "available",
    "method",
    "xyz_mm",
    "error_um",
    "confidence",
    "unverified",
    "reason",
)


@dataclass(frozen=True)
class ElectrodeSite:
    channel_id: int
    xyz_mm: tuple[float, float, float]
    label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "xyz_mm": list(self.xyz_mm),
            "label": self.label,
        }


@dataclass(frozen=True)
class LocalizationRequest:
    channel_ids: tuple[int, ...]
    electrode_xyz_mm: tuple[tuple[float, float, float], ...]
    event_ids: tuple[str, ...]
    snippets_uv: np.ndarray | None = None
    sample_rate_hz: float = 32_000.0
    t_sample: int = 0
    t0_sample: int = 0
    peak_channel: int | None = None
    ground_truth_xyz_mm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class LocalizationResult:
    available: bool
    reason: str
    xyz_mm: tuple[float, float, float] | None = None
    confidence: float | None = None
    method: str = "none"
    error_um: float | None = None
    unverified: bool = True
    score_profile: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": bool(self.available),
            "method": self.method,
            "xyz_mm": None if self.xyz_mm is None else [float(axis) for axis in self.xyz_mm],
            "error_um": None if self.error_um is None else float(self.error_um),
            "confidence": None if self.confidence is None else float(self.confidence),
            "unverified": True,
            "reason": self.reason,
            "score_profile": None if self.score_profile is None else list(self.score_profile),
        }


class LocalizationEngine(Protocol):
    def estimate(self, request: LocalizationRequest) -> LocalizationResult:
        ...


class NotImplementedLocalizer:
    """Placeholder so the UI can show a labeled empty localization pane."""

    method = "not_implemented"

    def estimate(self, request: LocalizationRequest) -> LocalizationResult:
        del request
        return LocalizationResult(
            available=False,
            reason="Localization not implemented",
            method=self.method,
        )


def empty_session_localization(reason: str = "Localization not implemented") -> dict[str, object]:
    return LocalizationResult(available=False, reason=reason, method="not_implemented").to_dict()


def _stable_event_id(label: str) -> int:
    digits = "".join(character if character.isdigit() else " " for character in label)
    parts = [int(part) for part in digits.split() if part]
    if parts:
        return int(parts[0])
    return 0


def resolve_edge_localizer(method: str = "amplitude") -> Any:
    from edge.localize import AmplitudeLocalizer, BeamformLocalizer, TemplateLocalizer
    from edge.localize.weighted import WeightedScanLocalizer

    engines = {"amplitude": AmplitudeLocalizer, "template": TemplateLocalizer,
               "beamform": BeamformLocalizer, "weighted_scan": WeightedScanLocalizer}
    if method not in engines:
        raise ValueError(f"unknown localization method {method}")
    return engines[method]()


def _result_from_edge(estimate: LocalizationEstimate, request: LocalizationRequest) -> LocalizationResult:
    xyz_mm: tuple[float, float, float] | None = None
    error_um = estimate.error_um
    gt_um = estimate.ground_truth_xyz_um
    if gt_um is None and request.ground_truth_xyz_mm is not None:
        gt_um = display_mm_to_probe_um(request.ground_truth_xyz_mm)
    if estimate.xyz_um is not None:
        xyz_mm = probe_um_to_display_mm(estimate.xyz_um)
        if gt_um is not None and in_probe_volume_um(gt_um):
            error_um = euclidean_um(estimate.xyz_um, gt_um)
        else:
            # Stylized whole-region sources are not scientific probe_um ground truth.
            error_um = None
    available = bool(estimate.available and xyz_mm is not None)
    reason = estimate.reason
    if available and not reason:
        reason = "Simplified amplitude localization (unverified)"
    return LocalizationResult(
        available=available,
        reason=reason,
        xyz_mm=xyz_mm,
        confidence=estimate.confidence,
        method=estimate.method,
        error_um=error_um,
        unverified=True,
    )


class EdgeLocalizerAdapter:
    """Convert viz millimetre electrodes and µV snippets into edge.types.Snippet."""

    def __init__(self, engine: Any | None = None, method: str = "amplitude") -> None:
        self._engine = engine if engine is not None else resolve_edge_localizer(method)
        self.method = getattr(self._engine, "method", "not_implemented")

    def estimate(self, request: LocalizationRequest) -> LocalizationResult:
        if request.snippets_uv is None:
            return NotImplementedLocalizer().estimate(request)
        if self._engine is None:
            return LocalizationResult(
                available=False,
                reason="Edge localizer not implemented",
                method="not_implemented",
            )
        try:
            voltages = np.asarray(request.snippets_uv, dtype=np.float64)
            if voltages.ndim != 2:
                return LocalizationResult(
                    available=False,
                    reason="Snippet voltages must be [samples, channels]",
                    method=self.method,
                )
            electrode_xyz_um = np.asarray(
                [display_mm_to_probe_um(xyz) for xyz in request.electrode_xyz_mm],
                dtype=np.float64,
            )
            gt_um = None
            if request.ground_truth_xyz_mm is not None:
                gt_um = display_mm_to_probe_um(request.ground_truth_xyz_mm)
            label = request.event_ids[0] if request.event_ids else "0"
            snippet = Snippet(
                event_id=_stable_event_id(label),
                t_sample=int(request.t_sample),
                t0_sample=int(request.t0_sample),
                sample_rate_hz=float(request.sample_rate_hz),
                channel_ids=tuple(int(channel) for channel in request.channel_ids),
                voltages_uv=voltages,
                electrode_xyz_um=electrode_xyz_um,
                peak_channel=request.peak_channel,
                ground_truth_xyz_um=gt_um,
            )
            estimate = self._engine.estimate(snippet)
            if isinstance(estimate, LocalizationResult):
                payload = estimate.to_dict()
                return LocalizationResult(
                    available=bool(payload["available"]),
                    reason=str(payload["reason"]),
                    xyz_mm=None if payload["xyz_mm"] is None else tuple(payload["xyz_mm"]),  # type: ignore[arg-type]
                    confidence=payload.get("confidence"),  # type: ignore[arg-type]
                    method=str(payload.get("method") or self.method),
                    error_um=payload.get("error_um"),  # type: ignore[arg-type]
                )
            if isinstance(estimate, LocalizationEstimate):
                result = _result_from_edge(estimate, request)
                if hasattr(self._engine, "score_profile"):
                    return LocalizationResult(**{**result.__dict__,
                        "score_profile": self._engine.score_profile(snippet)})
                return result
            return LocalizationResult(
                available=False,
                reason="Edge localizer returned an unsupported estimate type",
                method=self.method,
            )
        except Exception as exc:
            return LocalizationResult(
                available=False,
                reason=f"Edge localizer failed: {exc}",
                method=getattr(self._engine, "method", "not_implemented"),
            )


@dataclass
class ElectrodeArray:
    sites: list[ElectrodeSite] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "n_channels": len(self.sites),
            "sites": [site.to_dict() for site in self.sites],
        }


def default_probe_array() -> ElectrodeArray:
    """Approximate 32-site linear probe placed in the stylized visual cortex.

    Geometry is inspired by a 32-channel shank pitch, not a fabricated probe.
    """
    region_center = (0.0, -72.0, 8.0)
    pitch_mm = 0.025
    sites = []
    for channel in range(32):
        y = region_center[1] - 8.0 + channel * pitch_mm * 20.0
        sites.append(
            ElectrodeSite(
                channel_id=channel,
                xyz_mm=(region_center[0] + 4.0, y, region_center[2]),
                label=f"ch{channel:02d}",
            )
        )
    return ElectrodeArray(
        sites=sites,
        description="Stylized 32-channel linear probe in homologous visual cortex (approximate)",
    )
