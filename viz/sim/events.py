"""Versioned neural event records shared by the engine, API, and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = 1
COORDINATE_FRAMES = frozenset({"stylized_human", "allen_ccf_v3"})
POSITION_QUALITIES = frozenset({"simulated", "unit_ccf", "electrode_channel", "unknown"})
AMPLITUDE_UNITS = frozenset({"uv", "normalized"})
EVENT_KINDS = frozenset(
    {
        "synthetic_source",
        "neuropixels_spike",
        "electrode_activity",
        "detected_spike",
        "missed_spike",
        "localization_estimate",
    }
)
REQUIRED_FIELDS = (
    "schema_version",
    "event_id",
    "t_s",
    "xyz_mm",
    "coordinate_frame",
    "position_quality",
    "region_id",
    "amplitude",
    "amplitude_units",
    "source_id",
    "duration_ms",
    "kind",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class NeuralEvent:
    event_id: str
    t_s: float
    xyz_mm: tuple[float, float, float]
    region_id: str
    amplitude: float
    source_id: str
    kind: str
    duration_ms: float = 1.0
    coordinate_frame: str = "stylized_human"
    position_quality: str = "simulated"
    amplitude_units: str = "uv"
    channel_id: int | None = None
    ground_truth_event_id: str | None = None
    metadata: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        if not self.event_id:
            raise ValueError("event_id is required")
        if len(self.xyz_mm) != 3:
            raise ValueError("xyz_mm must have three coordinates")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError(f"unknown coordinate_frame {self.coordinate_frame}")
        if self.position_quality not in POSITION_QUALITIES:
            raise ValueError(f"unknown position_quality {self.position_quality}")
        if self.amplitude_units not in AMPLITUDE_UNITS:
            raise ValueError(f"unknown amplitude_units {self.amplitude_units}")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown kind {self.kind}")
        if self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["xyz_mm"] = [float(axis) for axis in self.xyz_mm]
        if payload["metadata"]:
            payload["metadata"] = _json_safe(payload["metadata"])
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NeuralEvent:
        missing = [name for name in REQUIRED_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"event missing fields: {missing}")
        xyz = payload["xyz_mm"]
        return cls(
            event_id=str(payload["event_id"]),
            t_s=float(payload["t_s"]),
            xyz_mm=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            region_id=str(payload["region_id"]),
            amplitude=float(payload["amplitude"]),
            source_id=str(payload["source_id"]),
            kind=str(payload["kind"]),
            duration_ms=float(payload["duration_ms"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            position_quality=str(payload["position_quality"]),
            amplitude_units=str(payload["amplitude_units"]),
            channel_id=None if payload.get("channel_id") is None else int(payload["channel_id"]),
            ground_truth_event_id=payload.get("ground_truth_event_id"),
            metadata=payload.get("metadata"),
            schema_version=int(payload["schema_version"]),
        )


@dataclass
class EventBatch:
    t_start_s: float
    t_end_s: float
    events: list[NeuralEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "events": [event.to_dict() for event in self.events],
        }
