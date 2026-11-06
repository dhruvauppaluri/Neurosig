"""Phase 5 floating-point detector variants for real AP recordings.

``P5Config()`` defaults reproduce the frozen ``car_4sigma_300_6000_negative``
detector exactly (verified by ``tests/test_phase5_detectors.py``). Each field is
one experiment axis. Nothing here touches the original modules.

Pipeline (all stages float64, samples are [time, channel]):
  1. causal 300-6000 Hz biquad cascade (same SOS as ``scripts.filter_iir``)
  2. common reference over channels on the *filtered* data
  3. per-channel baseline/sigma
  4. negative threshold crossing -> trough, per-channel refractory
  5. optional shape / amplitude gates, spatial artifact veto
  6. cross-channel merge
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

import numpy as np
from scipy import signal

from scripts.detect_spikes import MAD_NORMAL_SCALE, Event, estimate_noise
from scripts.filter_iir import IirConfig, design_sos


@dataclass(frozen=True)
class P5Config:
    sample_rate_hz: float = 30_000.0
    highpass_hz: float = 300.0
    lowpass_hz: float | None = 6_000.0
    # common reference: none | median | mean | trimmed (mean of middle 50 %)
    car: str = "median"
    # noise: mad_iter | mad | rms_clip | iqr | ema (causal running estimate)
    noise: str = "mad_iter"
    threshold_sigma: float = 4.0
    mad_clip_sigma: float = 6.0
    mad_iterations: int = 3
    ema_tau_s: float = 0.5
    # per-channel threshold floor in microvolts (0 disables)
    min_threshold_uv: float = 0.0
    peak_search_ms: float = 0.5
    refractory_ms: float = 1.0
    # shape gates (0 disables)
    min_width_samples: int = 0
    max_amplitude_sigma: float = 0.0
    # spatial artifact veto: fraction of channels firing within 0.2 ms
    artifact_channel_fraction: float = 0.5
    # merge: global (baseline anchored groups) | nms (time+space suppression)
    merge: str = "global"
    merge_ms: float = 0.5
    merge_radius_um: float = 0.0
    # spatial combination: x_c + w * sum(neighbours within radius), applied after the
    # common reference; weight 0 disables. Hardware cost: one small adder tree per channel.
    spatial_weight: float = 0.0
    spatial_radius_um: float = 40.0
    # constant timestamp shift in samples, added to every event (latency compensation)
    delay_comp_samples: int = 0

    def __post_init__(self) -> None:
        if self.car not in {"none", "median", "mean", "trimmed"}:
            raise ValueError(f"unknown car {self.car}")
        if self.noise not in {"mad_iter", "mad", "rms_clip", "iqr", "ema"}:
            raise ValueError(f"unknown noise {self.noise}")
        if self.merge not in {"global", "nms"}:
            raise ValueError(f"unknown merge {self.merge}")

    def samples(self, ms: float) -> int:
        return int(round(ms * self.sample_rate_hz / 1000.0))

    def replace(self, **kwargs) -> "P5Config":
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return asdict(self)


BASELINE = P5Config()


# ----------------------------------------------------------------- filtering
def filter_causal(voltages_uv: np.ndarray, cfg: P5Config) -> np.ndarray:
    iir = IirConfig(sample_rate_hz=cfg.sample_rate_hz, highpass_hz=cfg.highpass_hz,
                    lowpass_hz=cfg.lowpass_hz)
    return signal.sosfilt(design_sos(iir), np.asarray(voltages_uv, dtype=np.float64), axis=0)


def common_reference(filtered: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return filtered
    if method == "median":
        return filtered - np.median(filtered, axis=1, keepdims=True)
    if method == "mean":
        return filtered - np.mean(filtered, axis=1, keepdims=True)
    if method == "trimmed":
        n = filtered.shape[1]
        lo, hi = n // 4, n - n // 4
        ordered = np.sort(filtered, axis=1)[:, lo:hi]
        return filtered - ordered.mean(axis=1, keepdims=True)
    raise ValueError(method)


# --------------------------------------------------------------------- noise
def _mad_pass(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.median(x, axis=0)
    return base, np.median(np.abs(x - base), axis=0) / MAD_NORMAL_SCALE


def estimate_noise_variant(filtered: np.ndarray, cfg: P5Config) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel (baseline, sigma) for the chosen estimator."""
    x = filtered
    if cfg.noise == "mad":
        return estimate_noise(x)
    if cfg.noise == "mad_iter":
        base, sigma = estimate_noise(x)
        for _ in range(cfg.mad_iterations):
            nb, ns = np.empty_like(base), np.empty_like(sigma)
            for c in range(x.shape[1]):
                tr = x[:, c]
                keep = np.abs(tr - base[c]) <= cfg.mad_clip_sigma * sigma[c]
                if int(np.count_nonzero(keep)) < 32:
                    keep = np.ones(tr.shape[0], dtype=bool)
                b, s = estimate_noise(tr[keep][:, None])
                nb[c], ns[c] = b[0], s[0]
            base, sigma = nb, ns
        return base, sigma
    if cfg.noise == "rms_clip":
        base, sigma = estimate_noise(x)
        for _ in range(3):
            keep = np.abs(x - base) <= cfg.mad_clip_sigma * sigma
            n = np.maximum(keep.sum(axis=0), 1)
            base = np.where(keep, x, 0.0).sum(axis=0) / n
            sigma = np.sqrt(np.where(keep, (x - base) ** 2, 0.0).sum(axis=0) / n)
        return base, np.maximum(sigma, 1e-12)
    if cfg.noise == "iqr":
        q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0)
        return q50, (q75 - q25) / 1.3489795003921634
    if cfg.noise == "ema":
        # Causal running mean-absolute-deviation with outlier gating; the
        # per-sample sigma track is returned as a [time, channel] array by
        # ``ema_sigma_track``. Here return the final value for reporting.
        base, track = ema_sigma_track(x, cfg)
        return base, track[-1]
    raise ValueError(cfg.noise)


def ema_sigma_track(filtered: np.ndarray, cfg: P5Config) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-sample sigma: EMA of |x| gated at clip*sigma, seeded from the
    first 0.25 s median-MAD. 0.7979 converts mean|x| of a Gaussian to sigma."""
    x = filtered
    fs = cfg.sample_rate_hz
    seed_n = max(64, int(0.25 * fs))
    base = np.median(x[:seed_n], axis=0)
    sigma0 = np.median(np.abs(x[:seed_n] - base), axis=0) / MAD_NORMAL_SCALE
    alpha = 1.0 / (cfg.ema_tau_s * fs)
    track = np.empty_like(x)
    sigma = sigma0.copy()
    gate = cfg.mad_clip_sigma
    for t in range(x.shape[0]):
        d = np.abs(x[t] - base)
        upd = np.where(d <= gate * sigma, d / 0.7978845608, sigma)
        sigma = sigma + alpha * (upd - sigma)
        track[t] = sigma
    return base, track


# ----------------------------------------------------------------- detection
def detect_channel_events(
    filtered: np.ndarray,
    baseline: np.ndarray,
    sigma: np.ndarray | np.ndarray,
    cfg: P5Config,
    sigma_track: np.ndarray | None = None,
) -> list[Event]:
    """Per-channel negative threshold crossing with the baseline's refractory
    rule. ``sigma_track`` ([time, channel]) overrides ``sigma`` for adaptive
    thresholds. At default settings this equals ``scripts.detect_spikes.detect_candidates``.
    """
    search = cfg.samples(cfg.peak_search_ms)
    refractory = cfg.samples(cfg.refractory_ms)
    events: list[Event] = []
    for ch in range(filtered.shape[1]):
        trace = filtered[:, ch]
        if sigma_track is not None:
            thr_track = baseline[ch] - cfg.threshold_sigma * sigma_track[:, ch]
            if cfg.min_threshold_uv > 0:
                thr_track = np.minimum(thr_track, baseline[ch] - cfg.min_threshold_uv)
            below = trace < thr_track
            sig_at = sigma_track[:, ch]
        else:
            thr = baseline[ch] - cfg.threshold_sigma * sigma[ch]
            if cfg.min_threshold_uv > 0:
                thr = min(thr, baseline[ch] - cfg.min_threshold_uv)
            below = trace < thr
            sig_at = None
        crossings = np.flatnonzero(~below[:-1] & below[1:]) + 1
        last_peak = -refractory - 1
        for crossing in crossings:
            if crossing - last_peak < refractory:
                continue
            stop = min(len(trace), crossing + search + 1)
            peak = int(crossing + np.argmin(trace[crossing:stop]))
            if peak - last_peak < refractory:
                continue
            s = sig_at[peak] if sig_at is not None else sigma[ch]
            norm = float((baseline[ch] - trace[peak]) / s)
            events.append(Event(peak, ch, float(trace[peak]), norm))
            last_peak = peak
    return sorted(events, key=lambda e: (e.sample, e.channel))


def trough_width(trace: np.ndarray, peak: int, baseline: float, half_depth_factor: float = 0.5) -> int:
    """Samples contiguously below ``baseline - f*depth`` around the trough."""
    depth = baseline - trace[peak]
    level = baseline - half_depth_factor * depth
    lo = peak
    while lo > 0 and trace[lo - 1] < level:
        lo -= 1
    hi = peak
    while hi < len(trace) - 1 and trace[hi + 1] < level:
        hi += 1
    return hi - lo + 1


def shape_gate(events: list[Event], filtered: np.ndarray, baseline: np.ndarray, cfg: P5Config) -> tuple[list[Event], int]:
    if cfg.min_width_samples <= 0 and cfg.max_amplitude_sigma <= 0:
        return events, 0
    kept, dropped = [], 0
    for e in events:
        if cfg.max_amplitude_sigma > 0 and e.normalized_amplitude > cfg.max_amplitude_sigma:
            dropped += 1
            continue
        if cfg.min_width_samples > 0:
            if trough_width(filtered[:, e.channel], e.sample, baseline[e.channel]) < cfg.min_width_samples:
                dropped += 1
                continue
        kept.append(e)
    return kept, dropped


def spatial_artifact_veto(events: list[Event], n_channels: int, cfg: P5Config) -> tuple[list[Event], int]:
    """Identical to ``scripts.real_detector.reject_spatial_artifacts``."""
    if not events:
        return events, 0
    merge_samples = max(1, cfg.samples(0.2))
    ordered = sorted(events, key=lambda e: e.sample)
    kept: list[Event] = []
    rejected = 0
    i = 0
    need = max(2, int(np.ceil(cfg.artifact_channel_fraction * n_channels)))
    while i < len(ordered):
        start = ordered[i].sample
        group = [ordered[i]]
        i += 1
        while i < len(ordered) and ordered[i].sample - start <= merge_samples:
            group.append(ordered[i])
            i += 1
        if len({e.channel for e in group}) >= need:
            rejected += len(group)
            continue
        kept.extend(group)
    return kept, rejected


def merge_global(events: list[Event], cfg: P5Config) -> list[Event]:
    """Baseline: groups anchored on first event; keep strongest."""
    if not events:
        return []
    ms = cfg.samples(cfg.merge_ms)
    groups: list[list[Event]] = []
    for e in sorted(events, key=lambda c: c.sample):
        if not groups or e.sample - groups[-1][0].sample > ms:
            groups.append([e])
        else:
            groups[-1].append(e)
    return [max(g, key=lambda c: c.normalized_amplitude) for g in groups]


def merge_nms(events: list[Event], coordinates_um: np.ndarray, cfg: P5Config) -> list[Event]:
    """Strongest-first suppression of events within merge_ms and merge_radius_um.

    A radius <= 0 suppresses regardless of distance (global NMS). Distinct
    units on distant contacts survive when the radius is finite.
    """
    if not events:
        return []
    ms = cfg.samples(cfg.merge_ms)
    order = sorted(events, key=lambda e: -e.normalized_amplitude)
    kept: list[Event] = []
    for e in order:
        clash = False
        for k in kept:
            if abs(e.sample - k.sample) <= ms:
                if cfg.merge_radius_um <= 0:
                    clash = True
                    break
                d = float(np.linalg.norm(coordinates_um[e.channel] - coordinates_um[k.channel]))
                if d <= cfg.merge_radius_um:
                    clash = True
                    break
        if not clash:
            kept.append(e)
    return sorted(kept, key=lambda e: (e.sample, e.channel))


# ----------------------------------------------------------------------- run
def preprocess(voltages_uv: np.ndarray, cfg: P5Config, filtered_cache: np.ndarray | None = None) -> np.ndarray:
    filtered = filtered_cache if filtered_cache is not None else filter_causal(voltages_uv, cfg)
    return common_reference(filtered, cfg.car)


def spatial_matrix(coordinates_um: np.ndarray, weight: float, radius_um: float) -> np.ndarray:
    pos = np.asarray(coordinates_um, dtype=float)[:, :3]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    m = np.where((d <= radius_um) & (d > 0), weight, 0.0)
    np.fill_diagonal(m, 1.0)
    return m / np.sqrt((m ** 2).sum(axis=1, keepdims=True))


def run_detector(
    voltages_uv: np.ndarray,
    cfg: P5Config,
    coordinates_um: np.ndarray | None = None,
    filtered_cache: np.ndarray | None = None,
) -> dict:
    """Run the full variant pipeline. Returns events plus stage diagnostics."""
    t0 = time.perf_counter()
    x = np.asarray(voltages_uv, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("voltages must be a finite [samples, channels] array")
    n_ch = x.shape[1]
    ref = preprocess(x, cfg, filtered_cache)
    if cfg.spatial_weight > 0:
        if coordinates_um is None:
            raise ValueError("spatial combination needs coordinates")
        ref = ref @ spatial_matrix(coordinates_um, cfg.spatial_weight, cfg.spatial_radius_um).T
    track = None
    if cfg.noise == "ema":
        base, track = ema_sigma_track(ref, cfg)
        sigma = track[-1]
    else:
        base, sigma = estimate_noise_variant(ref, cfg)
    cands = detect_channel_events(ref, base, sigma, cfg, track)
    cands_gated, gated = shape_gate(cands, ref, base, cfg)
    cleaned, vetoed = spatial_artifact_veto(cands_gated, n_ch, cfg) if cfg.artifact_channel_fraction <= 1.0 else (cands_gated, 0)
    if cfg.merge == "global":
        merged = merge_global(cleaned, cfg)
    else:
        coords = coordinates_um if coordinates_um is not None else np.zeros((n_ch, 3))
        merged = merge_nms(cleaned, np.asarray(coords)[:, :3] if np.asarray(coords).shape[1] >= 3 else np.asarray(coords), cfg)
    if cfg.delay_comp_samples:
        merged = [Event(e.sample + cfg.delay_comp_samples, e.channel, e.voltage_uv, e.normalized_amplitude) for e in merged]
    return {
        "config": cfg.as_dict(),
        "events": merged,
        "candidates": cands,
        "n_candidates": len(cands),
        "n_shape_gated": gated,
        "n_artifact_vetoed": vetoed,
        "n_merged": len(merged),
        "reference": ref,
        "baseline": base,
        "sigma": sigma,
        "runtime_s": time.perf_counter() - t0,
    }
