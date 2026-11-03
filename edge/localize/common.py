"""Shared localization helpers. Errors are Euclidean distances in micrometers."""

from __future__ import annotations

import time

import numpy as np

from edge.coords import euclidean_um
from edge.snippets import channel_peak_amplitudes
from edge.types import LocalizationEstimate, Snippet


def l2_normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    safe = np.maximum(norm, 1e-18)
    return values / safe


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(np.asarray(a, dtype=np.float64).reshape(-1))
    b_n = l2_normalize(np.asarray(b, dtype=np.float64).reshape(-1))
    return float(np.dot(a_n, b_n))


def finish_estimate(
    *,
    method: str,
    xyz_um: tuple[float, float, float],
    snippet: Snippet,
    t_start: float,
    confidence: float,
    reason: str,
) -> LocalizationEstimate:
    xyz = (float(xyz_um[0]), float(xyz_um[1]), float(xyz_um[2]))
    gt = snippet.ground_truth_xyz_um
    error = None if gt is None else euclidean_um(xyz, gt)
    return LocalizationEstimate(
        available=True,
        method=method,
        xyz_um=xyz,
        ground_truth_xyz_um=gt,
        error_um=error,
        latency_s=time.perf_counter() - t_start,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        reason=reason,
        event_id=snippet.event_id,
        unverified=True,
    )


def electrode_com(snippet: Snippet, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    xyz = np.asarray(snippet.electrode_xyz_um, dtype=np.float64)
    if weights is None:
        weights = np.ones(xyz.shape[0], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)
    com = (w[:, None] * xyz).sum(axis=0) / np.sum(w)
    return float(com[0]), float(com[1]), float(com[2])


def snippet_amplitudes(snippet: Snippet) -> np.ndarray:
    return channel_peak_amplitudes(snippet)


def fold_linear_probe_ambiguity(
    xyz_um: tuple[float, float, float],
    electrode_xyz_um: np.ndarray,
) -> tuple[float, float, float]:
    """Map a monopole estimate onto the identifiable +x / z=0 half-plane.

    Contacts of ``linear_probe_um`` are collinear on y at x=z=0, so a quasi-static
    monopole is unchanged by rotation about the shank. When the array has no x/z
    span, report ``(sqrt(x^2+z^2), y, 0)``.
    """
    electrodes = np.asarray(electrode_xyz_um, dtype=np.float64).reshape(-1, 3)
    span_x = float(np.ptp(electrodes[:, 0])) if electrodes.size else 0.0
    span_z = float(np.ptp(electrodes[:, 2])) if electrodes.size else 0.0
    x, y, z = (float(v) for v in xyz_um)
    if span_x < 1e-6 and span_z < 1e-6:
        return (float(np.hypot(x, z)), y, 0.0)
    return (x, y, z)
