"""Spike-count and localization-derived feature vectors for edge decoding."""

from __future__ import annotations

import numpy as np

from edge.geometry import linear_probe_um

N_CHANNELS = 32
N_Y_BINS = 8
Y_MIN_UM = 0.0
Y_MAX_UM = 31 * 20.0  # linear_probe_um default pitch
SPATIAL_DIM = N_CHANNELS + N_Y_BINS + 3  # counts + occupancy + mean xyz


def probe_y_um(n_channels: int = N_CHANNELS) -> np.ndarray:
    return linear_probe_um(n_channels=n_channels).xyz_um()[:, 1]


def y_occupancy(
    y_um: np.ndarray,
    n_bins: int = N_Y_BINS,
    y_min: float = Y_MIN_UM,
    y_max: float = Y_MAX_UM,
) -> np.ndarray:
    occ = np.zeros(n_bins, dtype=np.float64)
    if y_um is None or len(y_um) == 0:
        return occ
    edges = np.linspace(y_min, y_max, n_bins + 1)
    idx = np.clip(np.digitize(np.asarray(y_um, dtype=np.float64), edges) - 1, 0, n_bins - 1)
    np.add.at(occ, idx, 1.0)
    return occ


def mean_xyz(points_um: np.ndarray) -> np.ndarray:
    if points_um is None or len(points_um) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(points_um, dtype=np.float64).reshape(-1, 3).mean(axis=0)


def baseline_counts(counts: np.ndarray) -> np.ndarray:
    row = np.asarray(counts, dtype=np.float64).reshape(-1)
    if row.size != N_CHANNELS:
        raise ValueError(f"baseline counts must have {N_CHANNELS} bins, got {row.size}")
    return row


def spatial_vector(counts: np.ndarray, estimated_xyz_um: np.ndarray) -> np.ndarray:
    """Concatenate unsigned-count features with localization-derived extras."""
    counts_f = baseline_counts(counts)
    xyz = np.asarray(estimated_xyz_um, dtype=np.float64).reshape(-1, 3) if len(estimated_xyz_um) else np.zeros((0, 3))
    occ = y_occupancy(xyz[:, 1] if len(xyz) else np.zeros(0))
    return np.concatenate([counts_f, occ, mean_xyz(xyz)])


def stack_baseline(count_rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(count_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != N_CHANNELS:
        raise ValueError(f"baseline stack must be [n, {N_CHANNELS}]")
    return rows


def stack_spatial(count_rows: np.ndarray, estimates: list[np.ndarray]) -> np.ndarray:
    rows = np.asarray(count_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != N_CHANNELS:
        raise ValueError(f"count stack must be [n, {N_CHANNELS}]")
    if len(estimates) != len(rows):
        raise ValueError("one estimate array required per example")
    out = np.empty((len(rows), SPATIAL_DIM), dtype=np.float64)
    for i, (counts, xyz) in enumerate(zip(rows, estimates)):
        out[i] = spatial_vector(counts, xyz)
    return out
