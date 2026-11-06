"""Matching and uncertainty summaries against an imperfect sorter reference.

Matching is optimal one-to-one assignment (minimum total |dt|) restricted to
pairs within the tolerance. It is used identically for every detector,
including the frozen baseline. ``legacy_match`` reproduces the earlier ordered
two-pointer matcher so published baseline numbers can be reproduced.

Sorter output is not ground truth; these numbers are coincidence rates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.detect_spikes import DetectorConfig, Event, match_events

TOLERANCES_MS = (0.5, 1.0, 2.0)


def match_optimal(det_samples: np.ndarray, ref_samples: np.ndarray, tol_samples: int,
                  allowed: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Return (det_index, ref_index, signed_dt) pairs. ``allowed`` optionally
    masks [det, ref] pairs (used for channel-consistent matching)."""
    d = np.asarray(det_samples, dtype=np.int64)
    r = np.asarray(ref_samples, dtype=np.int64)
    if len(d) == 0 or len(r) == 0:
        return []
    diff = d[:, None] - r[None, :]
    ok = np.abs(diff) <= tol_samples
    if allowed is not None:
        ok &= allowed
    if not ok.any():
        return []
    # restrict to rows/cols with at least one candidate pair to keep the LSA small
    rows = np.flatnonzero(ok.any(axis=1))
    cols = np.flatnonzero(ok.any(axis=0))
    sub_ok = ok[np.ix_(rows, cols)]
    big = 10 * (tol_samples + 1) * (len(rows) + len(cols) + 1)
    cost = np.where(sub_ok, np.abs(diff[np.ix_(rows, cols)]), big).astype(np.float64)
    ri, ci = linear_sum_assignment(cost)
    out = []
    for a, b in zip(ri, ci):
        if sub_ok[a, b]:
            out.append((int(rows[a]), int(cols[b]), int(diff[rows[a], cols[b]])))
    return out


def window_metrics(events: Sequence[Event], ref_times_s: np.ndarray, fs: float,
                   duration_s: float, tolerance_ms: float,
                   ref_channels: np.ndarray | None = None,
                   coordinates_um: np.ndarray | None = None,
                   channel_radius_um: float | None = None) -> dict:
    tol = int(round(tolerance_ms * fs / 1000.0))
    det = np.array([e.sample for e in events], dtype=np.int64)
    ref = np.rint(np.asarray(ref_times_s) * fs).astype(np.int64)
    allowed = None
    if channel_radius_um is not None and ref_channels is not None and coordinates_um is not None and len(det) and len(ref):
        dch = np.array([e.channel for e in events], dtype=int)
        pos = np.asarray(coordinates_um)[:, :3]
        dist = np.linalg.norm(pos[dch][:, None, :] - pos[np.asarray(ref_channels, dtype=int)][None, :, :], axis=2)
        allowed = dist <= channel_radius_um
    pairs = match_optimal(det, ref, tol, allowed)
    tp = len(pairs)
    fp = len(det) - tp
    fn = len(ref) - tp
    prec = tp / len(det) if len(det) else 0.0
    rec = tp / len(ref) if len(ref) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dt_ms = np.array([p[2] for p in pairs], dtype=float) / fs * 1000.0
    return {
        "tolerance_ms": tolerance_ms, "n_detected": int(len(det)), "n_reference": int(len(ref)),
        "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
        "false_detections_per_s": fp / duration_s,
        "timing_error_ms": dt_ms.tolist(),
        "matched_det_idx": [p[0] for p in pairs], "matched_ref_idx": [p[1] for p in pairs],
    }


def legacy_match(events: Sequence[Event], ref_times_s: np.ndarray, fs: float, tolerance_ms: float) -> dict:
    cfg = DetectorConfig(sample_rate_hz=fs, match_tolerance_ms=tolerance_ms)
    _, m = match_events(list(events), np.asarray(ref_times_s), cfg)
    return m


def pooled(per_window: Sequence[dict]) -> dict:
    """Micro-averaged counts and macro-averaged window recall/F1."""
    tp = sum(w["tp"] for w in per_window)
    fp = sum(w["fp"] for w in per_window)
    fn = sum(w["fn"] for w in per_window)
    nd = tp + fp
    nr = tp + fn
    prec = tp / nd if nd else 0.0
    rec = tp / nr if nr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_windows": len(per_window), "tp": tp, "fp": fp, "fn": fn,
        "n_detected": nd, "n_reference": nr,
        "precision": prec, "recall": rec, "f1": f1,
        "macro_recall": float(np.mean([w["recall"] for w in per_window])) if per_window else 0.0,
        "macro_f1": float(np.mean([w["f1"] for w in per_window])) if per_window else 0.0,
    }


def bootstrap_ci(per_window: Sequence[dict], stat: str = "f1", n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Cluster bootstrap over windows (the independent unit), pooled counts."""
    rng = np.random.default_rng(seed)
    n = len(per_window)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(pooled([per_window[i] for i in idx])[stat])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[dict], b: Sequence[dict], stat: str = "f1", n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Window-paired difference (a - b) with a 95 % cluster-bootstrap interval."""
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "ci": [float("nan"), float("nan")], "fraction_windows_a_better": float("nan")}
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(pooled([a[i] for i in idx])[stat] - pooled([b[i] for i in idx])[stat])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    better = float(np.mean([x[stat] > y[stat] for x, y in zip(a, b)]))
    return {"diff": pooled(a)[stat] - pooled(b)[stat], "ci": [float(lo), float(hi)],
            "fraction_windows_a_better": better}


def timing_summary(errors_ms: Sequence[float]) -> dict:
    e = np.asarray(errors_ms, dtype=float)
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median_ms": float(np.median(e)), "mean_ms": float(e.mean()),
            "mad_ms": float(np.median(np.abs(e - np.median(e)))),
            "p5_ms": float(np.percentile(e, 5)), "p95_ms": float(np.percentile(e, 95)),
            "mean_abs_ms": float(np.mean(np.abs(e)))}
