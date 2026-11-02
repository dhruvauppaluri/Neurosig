"""Interfaces for future spatial source localization. No algorithm is implemented."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


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


@dataclass(frozen=True)
class LocalizationResult:
    available: bool
    reason: str
    xyz_mm: tuple[float, float, float] | None = None
    confidence: float | None = None
    method: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "xyz_mm": None if self.xyz_mm is None else list(self.xyz_mm),
            "confidence": self.confidence,
            "method": self.method,
            "unverified": True,
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
