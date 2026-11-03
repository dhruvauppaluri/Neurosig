"""Detect spatially overlapping sources and refuse a single-point estimate.

A quasi-static monopole cannot unmix two nearby neurons. When the aligned
trough footprint has two strong, well-separated peaks along the probe,
localization should fail closed instead of reporting a biased centroid.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from edge.localize.common import aligned_trough_amplitudes
from edge.types import Snippet


def spatial_overlap_conflict(
    snippet: Snippet,
    *,
    min_peak_separation_um: float = 80.0,
    second_peak_fraction: float = 0.85,
) -> tuple[bool, str]:
    """Return True when two strong spatial peaks are separated along the shank."""
    amps = aligned_trough_amplitudes(snippet)
    electrodes = np.asarray(snippet.electrode_xyz_um, dtype=np.float64)
    if amps.size < 4 or electrodes.shape[0] != amps.size:
        return False, ""
    order = np.argsort(electrodes[:, 1])
    y = electrodes[order, 1]
    profile = np.asarray(amps[order], dtype=np.float64)
    if profile.size >= 5:
        profile = np.convolve(profile, np.array([0.25, 0.5, 0.25]), mode="same")
    peak = float(np.max(profile))
    if peak <= 0.0:
        return False, ""
    pitches = np.diff(y)
    pitch = float(np.median(pitches[pitches > 0])) if np.any(pitches > 0) else 20.0
    distance = max(2, int(round(min_peak_separation_um / max(pitch, 1.0))))
    peaks, _props = find_peaks(profile, height=second_peak_fraction * peak, distance=distance)
    if len(peaks) < 2:
        return False, ""
    ys = [float(y[i]) for i in peaks]
    return True, (
        f"bimodal trough footprint at {ys[0]:.1f} µm and {ys[1]:.1f} µm; "
        "refusing a single-source estimate"
    )
