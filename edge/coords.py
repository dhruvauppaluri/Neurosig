"""Affine map between scientific probe micrometers and stylized viewer millimetres.

The viewer probe in viz.sim.localization.default_probe_array is a display
placement, not a fabricated implant. Neuropixels CCF coordinates must not pass
through this transform.
"""

from __future__ import annotations

import numpy as np

# Matches viz.sim.localization.default_probe_array contact 0 and 0.5 mm pitch.
DISPLAY_ORIGIN_MM = (4.0, -80.0, 8.0)
DISPLAY_SHANK_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
SCIENTIFIC_PITCH_UM = 20.0
DISPLAY_PITCH_MM = 0.5
UM_PER_MM = 1000.0
SHANK_SCALE = DISPLAY_PITCH_MM / SCIENTIFIC_PITCH_UM  # mm per µm along the shank


def probe_um_to_display_mm(xyz_um: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    x_um, y_um, z_um = (float(axis) for axis in np.asarray(xyz_um, dtype=np.float64).reshape(3))
    return (
        DISPLAY_ORIGIN_MM[0] + x_um / UM_PER_MM,
        DISPLAY_ORIGIN_MM[1] + y_um * SHANK_SCALE,
        DISPLAY_ORIGIN_MM[2] + z_um / UM_PER_MM,
    )


def display_mm_to_probe_um(xyz_mm: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    x_mm, y_mm, z_mm = (float(axis) for axis in np.asarray(xyz_mm, dtype=np.float64).reshape(3))
    return (
        (x_mm - DISPLAY_ORIGIN_MM[0]) * UM_PER_MM,
        (y_mm - DISPLAY_ORIGIN_MM[1]) / SHANK_SCALE,
        (z_mm - DISPLAY_ORIGIN_MM[2]) * UM_PER_MM,
    )


def batch_probe_um_to_display_mm(points_um: np.ndarray) -> np.ndarray:
    points = np.asarray(points_um, dtype=np.float64)
    out = np.empty_like(points)
    for i, point in enumerate(points.reshape(-1, 3)):
        out.reshape(-1, 3)[i] = probe_um_to_display_mm(point)
    return out


def euclidean_um(
    a: tuple[float, float, float] | np.ndarray,
    b: tuple[float, float, float] | np.ndarray,
) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def in_probe_volume_um(
    xyz_um: tuple[float, float, float] | np.ndarray,
    *,
    n_channels: int = 32,
    pitch_um: float = SCIENTIFIC_PITCH_UM,
    radial_um: float = 150.0,
    y_pad_um: float = 40.0,
) -> bool:
    """True when a point is near the scientific linear probe, not a whole-brain display coordinate."""
    x_um, y_um, z_um = (float(axis) for axis in np.asarray(xyz_um, dtype=np.float64).reshape(3))
    y_max = (n_channels - 1) * pitch_um
    return abs(x_um) <= radial_um and abs(z_um) <= radial_um and -y_pad_um <= y_um <= y_max + y_pad_um
