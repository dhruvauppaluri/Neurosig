"""Stylized human-brain region volumes for visualization (not an atlas)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RegionVolume:
    region_id: str
    label: str
    center_mm: tuple[float, float, float]
    radii_mm: tuple[float, float, float]
    color: str
    hemisphere: str = "both"
    homologous_mouse: str | None = None
    description: str = ""

    def contains(self, xyz_mm: tuple[float, float, float], margin: float = 1.0) -> bool:
        dx = (xyz_mm[0] - self.center_mm[0]) / (self.radii_mm[0] * margin)
        dy = (xyz_mm[1] - self.center_mm[1]) / (self.radii_mm[1] * margin)
        dz = (xyz_mm[2] - self.center_mm[2]) / (self.radii_mm[2] * margin)
        return (dx * dx + dy * dy + dz * dz) <= 1.0

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        radius = rng.random() ** (1.0 / 3.0)
        point = np.array(self.center_mm) + np.array(self.radii_mm) * direction * radius
        return (float(point[0]), float(point[1]), float(point[2]))

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "label": self.label,
            "center_mm": list(self.center_mm),
            "radii_mm": list(self.radii_mm),
            "color": self.color,
            "hemisphere": self.hemisphere,
            "homologous_mouse": self.homologous_mouse,
            "description": self.description,
        }


# Approximate adult-human extents in millimetres, origin at commissural midpoint.
# These ellipsoids are visualization aids, not FreeSurfer or MNI labels.
REGIONS: dict[str, RegionVolume] = {
    "frontal_left": RegionVolume(
        "frontal_left",
        "Left frontal",
        (-28.0, 42.0, 18.0),
        (32.0, 38.0, 28.0),
        "#4cc9f0",
        "left",
        description="Stylized left frontal volume",
    ),
    "frontal_right": RegionVolume(
        "frontal_right",
        "Right frontal",
        (28.0, 42.0, 18.0),
        (32.0, 38.0, 28.0),
        "#4895ef",
        "right",
        description="Stylized right frontal volume",
    ),
    "motor_left": RegionVolume(
        "motor_left",
        "Left motor",
        (-32.0, 4.0, 42.0),
        (18.0, 16.0, 16.0),
        "#80ffdb",
        "left",
    ),
    "motor_right": RegionVolume(
        "motor_right",
        "Right motor",
        (32.0, 4.0, 42.0),
        (18.0, 16.0, 16.0),
        "#64dfdf",
        "right",
    ),
    "parietal_left": RegionVolume(
        "parietal_left",
        "Left parietal",
        (-28.0, -28.0, 36.0),
        (26.0, 24.0, 22.0),
        "#b5179e",
        "left",
    ),
    "parietal_right": RegionVolume(
        "parietal_right",
        "Right parietal",
        (28.0, -28.0, 36.0),
        (26.0, 24.0, 22.0),
        "#7209b7",
        "right",
    ),
    "temporal_left": RegionVolume(
        "temporal_left",
        "Left temporal",
        (-52.0, -8.0, -18.0),
        (22.0, 32.0, 18.0),
        "#f72585",
        "left",
    ),
    "temporal_right": RegionVolume(
        "temporal_right",
        "Right temporal",
        (52.0, -8.0, -18.0),
        (22.0, 32.0, 18.0),
        "#ff6b6b",
        "right",
    ),
    "visual_cortex": RegionVolume(
        "visual_cortex",
        "Visual / occipital cortex",
        (0.0, -72.0, 8.0),
        (28.0, 18.0, 20.0),
        "#ffe066",
        "both",
        homologous_mouse="VISp",
        description=(
            "Stylized occipital volume used to display mouse visual-cortex "
            "Neuropixels events. Not human V1 anatomy."
        ),
    ),
    "hippocampus_left": RegionVolume(
        "hippocampus_left",
        "Left hippocampus",
        (-28.0, -18.0, -12.0),
        (12.0, 18.0, 8.0),
        "#c77dff",
        "left",
    ),
    "hippocampus_right": RegionVolume(
        "hippocampus_right",
        "Right hippocampus",
        (28.0, -18.0, -12.0),
        (12.0, 18.0, 8.0),
        "#9d4edd",
        "right",
    ),
    "thalamus": RegionVolume(
        "thalamus",
        "Thalamus",
        (0.0, -8.0, 4.0),
        (14.0, 12.0, 10.0),
        "#ffd166",
        "both",
    ),
}

MOUSE_AREA_TO_REGION = {
    "VISp": "visual_cortex",
    "VISl": "visual_cortex",
    "VISal": "visual_cortex",
    "VISpm": "visual_cortex",
    "VISam": "visual_cortex",
    "VISrl": "visual_cortex",
    "VIS": "visual_cortex",
}

NEUROPIXELS_ALLOWED_REGIONS = frozenset({"visual_cortex"})


def get_region(region_id: str) -> RegionVolume:
    try:
        return REGIONS[region_id]
    except KeyError as exc:
        raise KeyError(f"unknown stylized region {region_id}") from exc


def map_mouse_area(acronym: str) -> str:
    key = acronym.strip()
    if key not in MOUSE_AREA_TO_REGION:
        raise ValueError(
            f"{acronym!r} is not a visual-area acronym supported for Neuropixels display"
        )
    return MOUSE_AREA_TO_REGION[key]


def region_catalog() -> list[dict[str, object]]:
    return [region.to_dict() for region in REGIONS.values()]


def lerp_in_region(
    region: RegionVolume,
    u: float,
    v: float,
    w: float,
) -> tuple[float, float, float]:
    """Map unit-cube coordinates in [0, 1] into the ellipsoid (stable placement)."""
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0)
    r = w ** (1.0 / 3.0)
    x = region.center_mm[0] + region.radii_mm[0] * r * math.sin(phi) * math.cos(theta)
    y = region.center_mm[1] + region.radii_mm[1] * r * math.sin(phi) * math.sin(theta)
    z = region.center_mm[2] + region.radii_mm[2] * r * math.cos(phi)
    return (x, y, z)
