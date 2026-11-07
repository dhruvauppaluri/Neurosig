"""Overlap benchmark (Goal B): singles, pairs, triples with known sources and a predefined resolvable subset.

Resolvable pair (fixed before any evaluation): y-separation >= 60 um, both sources' peak-channel SNR >= 6,
amplitude ratio (weaker/stronger) >= 1/3. Scoring is one-to-one (Hungarian) in the folded (rho, y) frame with a
60 um gate; missed truths are charged their distance to the nearest estimate and counted as misses, extra
estimates are counted as extras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from edge.benchmark_v3 import _array
from edge.forward import PointSource, predicted_amplitude_pattern, render_recording
from edge.snippets import extract_snippet
from edge.types import Snippet

GATE_UM = 60.0
SEEDS = {"dev": 20268101, "select": 20268102, "final": 20268103}
NOISE = (2.0, 5.0, 10.0, 25.0)


@dataclass(frozen=True)
class OverlapEvent:
    event_id: int
    pool: str
    kind: str               # single | pair | triple
    noise_uv: float
    truths_rho_y: tuple     # ((rho, y), ...) folded frame
    amps_uv: tuple
    snr: tuple
    dt_samples: tuple
    sep_um: float
    ratio: float
    resolvable: bool
    difficult: bool
    snippet: Snippet


def _folded(p):
    return (float(np.hypot(p[0], p[2])), float(p[1]))


def make_overlap_events(pool: str, n_per_noise: int = 40) -> list[OverlapEvent]:
    array = _array("linear")
    E = array.xyz_um()
    rng = np.random.default_rng(SEEDS[pool])
    events: list[OverlapEvent] = []
    for noise in NOISE:
        for kind in ("single", "pair", "pair", "triple"):  # pairs weighted 2x
            for _ in range(max(1, n_per_noise // 2) if kind != "pair" else n_per_noise):
                k = {"single": 1, "pair": 2, "triple": 3}[kind]
                pts, amps, dts = [], [], []
                y0 = float(rng.uniform(100, 540))
                for j in range(k):
                    if j == 0:
                        y = y0
                    else:
                        y = float(np.clip(y0 + rng.choice([-1, 1]) * rng.uniform(20, 150) * j, 30, 610))
                    pts.append((float(rng.uniform(12, 70)), y, float(rng.uniform(-25, 25))))
                    amps.append(float(rng.uniform(8, 16) if j == 0 else rng.uniform(8, 16) * rng.uniform(0.35, 1.0)))
                    dts.append(0 if j == 0 else int(rng.choice([-1, 1]) * rng.integers(0, 13)))
                sources = [PointSource(64 + d, p, a) for p, a, d in zip(pts, amps, dts)]
                trace = render_recording(sources, array, 32000.0, noise, int(rng.integers(0, 2**31)), 160)
                snr = tuple(float(a * np.max(predicted_amplitude_pattern(p, E)) / noise) for p, a in zip(pts, amps))
                sep = abs(pts[0][1] - pts[1][1]) if k >= 2 else float("nan")
                ratio = min(amps[:2]) / max(amps[:2]) if k >= 2 else float("nan")
                resolvable = bool(k == 2 and sep >= 60 and min(snr) >= 6 and ratio >= 1 / 3)
                difficult = bool(resolvable and abs(dts[1]) <= 4 and sep <= 120)
                eid = len(events)
                snip = extract_snippet(trace, 64, array, event_id=eid, ground_truth_xyz_um=pts[0])
                events.append(OverlapEvent(eid, pool, kind, noise, tuple(_folded(p) for p in pts), tuple(amps), snr,
                                           tuple(dts), sep, ratio, resolvable, difficult, snip))
    return events


def assign(truths: np.ndarray, est: np.ndarray, gate: float = GATE_UM) -> dict:
    """One-to-one assignment. truths [T,2], est [E,2] (E may be 0)."""
    T = truths.shape[0]
    if est.shape[0] == 0:
        return {"matched": 0, "missed": T, "extra": 0, "matched_err": [], "event_err": float("nan")}
    D = np.linalg.norm(truths[:, None] - est[None], axis=2)
    r, c = linear_sum_assignment(D)
    ok = D[r, c] <= gate
    matched_err = D[r, c][ok].tolist()
    matched_truth = set(r[ok].tolist())
    per_truth = np.empty(T)
    for t in range(T):
        per_truth[t] = D[t, c[list(r).index(t)]] if t in r else D[t].min()
    # unmatched truths (beyond gate or no partner) charged to the nearest estimate
    for t in range(T):
        if t not in matched_truth:
            per_truth[t] = D[t].min()
    used = {int(c[i]) for i in range(len(r)) if ok[i]}
    return {"matched": int(ok.sum()), "missed": int(T - ok.sum()), "extra": int(est.shape[0] - len(used)),
            "matched_err": matched_err, "event_err": float(per_truth.mean())}


def evaluate_methods(events: list[OverlapEvent], methods: dict) -> dict:
    """methods: name -> callable(snippet) -> (est [E,2] folded array (E may be 0), rejected: bool)."""
    out = {}
    for name, fn in methods.items():
        recs = []
        for ev in events:
            est, rejected = fn(ev.snippet)
            a = assign(np.asarray(ev.truths_rho_y), np.asarray(est).reshape(-1, 2))
            recs.append({"n_est": int(np.asarray(est).reshape(-1, 2).shape[0]), "rejected": bool(rejected), **a})
        out[name] = recs
    return out


def summarize_overlap(events: list[OverlapEvent], recs: list[dict]) -> dict:
    def sel(f):
        return [i for i, e in enumerate(events) if f(e)]
    res = sel(lambda e: e.resolvable)
    pairs = sel(lambda e: e.kind == "pair")
    singles = sel(lambda e: e.kind == "single")
    diff = sel(lambda e: e.difficult)
    triples = sel(lambda e: e.kind == "triple")
    def rate(idx, f):
        return float(np.mean([f(recs[i]) for i in idx])) if idx else None
    corr = [i for i in res if recs[i]["n_est"] == 2 and recs[i]["matched"] == 2]
    matched_err = [x for i in corr for x in recs[i]["matched_err"]]
    diff_err = [recs[i]["event_err"] for i in diff if np.isfinite(recs[i]["event_err"])]
    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else None
    return {
        "n": {"events": len(events), "resolvable_pairs": len(res), "all_pairs": len(pairs), "singles": len(singles),
              "difficult": len(diff), "triples": len(triples)},
        "two_source_recall_resolvable": rate(res, lambda r: r["n_est"] >= 2 and r["matched"] == 2 and not r["rejected"]),
        "two_source_recall_all_pairs": rate(pairs, lambda r: r["n_est"] >= 2 and r["matched"] == 2 and not r["rejected"]),
        "false_two_source_rate_singles": rate(singles, lambda r: r["n_est"] >= 2),
        "rejection_rate_all": rate(list(range(len(events))), lambda r: r["rejected"]),
        "rejection_rate_resolvable": rate(res, lambda r: r["rejected"]),
        "per_source_median_err_correct_pairs_um": q(matched_err, 50),
        "per_source_p90_err_correct_pairs_um": q(matched_err, 90),
        "n_correct_pairs": len(corr),
        "difficult_event_err_median_um": q(diff_err, 50), "difficult_event_err_p90_um": q(diff_err, 90),
        "all_pairs_event_err_p90_um": q([recs[i]["event_err"] for i in pairs if np.isfinite(recs[i]["event_err"])], 90),
        "missed_per_pair": rate(pairs, lambda r: r["missed"]), "extra_per_pair": rate(pairs, lambda r: r["extra"]),
        "triple_mean_sources_found": rate(triples, lambda r: r["n_est"]),
        "triple_missed_per_event": rate(triples, lambda r: r["missed"]),
    }
