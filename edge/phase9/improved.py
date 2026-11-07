"""Snippet conditioners and wrapped engines for Phase 9. Original edge/localize engines are never modified."""

from __future__ import annotations

from scipy.ndimage import median_filter
import numpy as np

from edge.types import LocalizationEstimate, Snippet


def _rebuild(sn: Snippet, v: np.ndarray) -> Snippet:
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v,
                   sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)


def edge_noise(v: np.ndarray, n_pre: int) -> np.ndarray:
    n = max(6, min(n_pre - 4, v.shape[0] // 3))
    e = np.concatenate([v[:n], v[-n:]], axis=0)
    mad = np.median(np.abs(e - np.median(e, axis=0)), axis=0) / 0.6745
    return np.maximum(mad, 1e-9)


def dead_channels(v: np.ndarray, n_pre: int) -> np.ndarray:
    s = edge_noise(v, n_pre)
    ref = np.median(s)
    return (s <= 1e-6) | (s < 0.05 * ref)


def interpolate_dead(v: np.ndarray, y: np.ndarray, dead: np.ndarray) -> np.ndarray:
    if not dead.any() or dead.all():
        return v
    out = v.copy()
    live = np.flatnonzero(~dead)
    order = np.argsort(y[live])
    for c in np.flatnonzero(dead):
        yy = y[live][order]
        for t in range(v.shape[0]):
            out[t, c] = np.interp(y[c], yy, v[t, live][order])
    return out


def condition(sn: Snippet, *, baseline=False, car=False, tmedian=False, tmean=False, tproj=False, noise_norm=False,
              interp_dead=False, clip_gain=(0.5, 2.0), tproj_half=3, tproj_top=6) -> Snippet:
    v = np.asarray(sn.voltages_uv, float).copy()
    n_pre = sn.n_pre
    if interp_dead:
        v = interpolate_dead(v, np.asarray(sn.electrode_xyz_um)[:, 1], dead_channels(v, n_pre))
    if baseline:
        k = max(5, n_pre - 12)
        v -= np.median(v[:k], axis=0, keepdims=True)
    if car:
        v -= np.median(v, axis=1, keepdims=True)
    if tmedian:
        v = median_filter(v, size=(3, 1), mode="nearest")
    if tmean:
        k = np.array([0.25, 0.5, 0.25])
        v = np.apply_along_axis(lambda x: np.convolve(np.pad(x, 1, mode="edge"), k, mode="valid"), 0, v)
    if tproj:
        # amplitude at the trough sample = projection on the common temporal waveform of the strongest channels
        lo, hi = max(n_pre - tproj_half, 0), min(n_pre + tproj_half + 1, v.shape[0])
        seg = v[lo:hi]
        top = np.argsort(np.sum(seg ** 2, axis=0))[::-1][:tproj_top]
        u, sv, vt = np.linalg.svd(seg[:, top], full_matrices=False)
        w = u[:, 0]
        if float(w @ seg[:, top].sum(axis=1)) > 0:
            w = -w                      # trough orientation: negative-going
        amp = w @ seg                     # per-channel projection; w is unit-norm, trough orientation negative
        scale = float(np.sum(np.abs(w)) / np.sqrt(len(w)))   # keeps amplitude on the single-sample scale
        v = v.copy()
        v[n_pre] = amp / max(scale, 1e-9) / np.sqrt(len(w))
    if noise_norm:
        s = edge_noise(v, n_pre)
        g = np.clip(np.median(s) / s, *clip_gain)
        v = v * g[None, :]
    return _rebuild(sn, v)


class ConditionedLocalizer:
    """Apply a fixed conditioner before an existing engine."""

    def __init__(self, engine, **options) -> None:
        self.engine = engine
        self.options = options
        self.method = getattr(engine, "method", "engine")

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        return self.engine.estimate(condition(snippet, **self.options))


def conditioned_factory(base_factory, **options):
    def factory(array):
        return {k: ConditionedLocalizer(e, **options) for k, e in base_factory(array).items()}
    return factory


# ---- noise-aware overlap rule and robust scan v2 -------------------------------------------------------------
from scipy.signal import find_peaks  # noqa: E402

from edge.localize.common import aligned_trough_amplitudes, finish_unavailable  # noqa: E402
from edge.localize.robust import RobustScanLocalizer  # noqa: E402


def noise_aware_overlap(snippet: Snippet, *, min_peak_separation_um=80.0, second_peak_fraction=0.85,
                        min_snr=4.0) -> tuple[bool, str]:
    """Like edge.localize.overlap.spatial_overlap_conflict, but a peak only counts if it clears ``min_snr`` noise
    sigmas (single-sample noise estimated from the snippet edges). Noise-driven second peaks no longer reject."""
    amps = aligned_trough_amplitudes(snippet)
    xyz = np.asarray(snippet.electrode_xyz_um, float)
    if amps.size < 4:
        return False, ""
    sigma = float(np.median(edge_noise(np.asarray(snippet.voltages_uv, float), snippet.n_pre)))
    order = np.argsort(xyz[:, 1])
    y, prof = xyz[order, 1], np.asarray(amps[order], float)
    if prof.size >= 5:
        prof = np.convolve(prof, [0.25, 0.5, 0.25], mode="same")
        sigma = sigma / np.sqrt(1.5 * 1.0) * 0.8          # approx std of the smoothed noise
    peak = float(prof.max())
    if peak < min_snr * sigma:
        return False, ""
    d = np.diff(y)
    pitch = float(np.median(d[d > 0])) if np.any(d > 0) else 20.0
    dist = max(2, int(round(min_peak_separation_um / max(pitch, 1.0))))
    peaks, _ = find_peaks(prof, height=max(second_peak_fraction * peak, min_snr * sigma), distance=dist)
    if len(peaks) < 2:
        return False, ""
    return True, f"noise-aware bimodal footprint at {y[peaks[0]]:.0f} and {y[peaks[1]]:.0f} um"


class RobustScanV2(RobustScanLocalizer):
    method = "robust_scan_v2"

    def __init__(self, *a, min_snr: float = 4.0, **kw):
        kw["fail_closed_on_overlap"] = False
        super().__init__(*a, **kw)
        self.min_snr = min_snr

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        import time
        t0 = time.perf_counter()
        conflict, reason = noise_aware_overlap(snippet, min_snr=self.min_snr)
        if conflict:
            return finish_unavailable(method=self.method, snippet=snippet, t_start=t0, reason=reason)
        return super().estimate(snippet)
