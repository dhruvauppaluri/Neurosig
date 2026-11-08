"""Phase 10 overlap stress tests on real-like synthetic data (known truth), not run in Phase 9:

* partial temporal overlap sweep (|dt| = 0, 1-3, 4-9, 10-20 samples) and separation sweep,
* three-source events (with the frozen K<=2 pipeline, and an optional greedy K=3 extension),
* electrode dropout during overlap (known mask / unknown zeroed contacts), singles as negative controls.

Uses the frozen Phase 9 overlap configuration (line source, lambda 20 um, rank 4, common-mode nuisance projection,
baseline + dead-channel interpolation, calibration from reports/phase9/overlap_uncertainty_final.json). Seed pools
are new (never used before) and the resolvable definition is the Phase 9 one, fixed in code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from edge.phase8.multisource import _gain
from edge.phase8.pipeline import Calibration, Phase8Localizer
from edge.phase9.improved import condition, dead_channels
from edge.phase9.synth_real import FS, _render, reference_array
from edge.snippets import extract_snippet

ROOT = Path(__file__).resolve().parents[2]
SEEDS = {"p10dev": 20271001, "p10test": 20271002, "p10confirm": 20271004}
OV_COND = dict(baseline=True, interp_dead=True)
GATE_UM = 60.0


def frozen_calibration() -> Calibration:
    d = json.loads((ROOT / "reports/phase9/overlap_uncertainty_final.json").read_text())["calibration"]
    return Calibration(**d)


def localizer(array, cal=None) -> Phase8Localizer:
    return Phase8Localizer(array, cal or frozen_calibration(), model="line_source", lambda_um=20.0, rank=4,
                           max_x_um=90.0, y_margin_um=40.0, nuisance_cm=True)


@dataclass(frozen=True)
class StressEvent:
    kind: str                 # single | pair | triple
    noise_uv: float
    sources: tuple            # (x, y, z, amp, t)
    dt: int                   # |dt| between first two sources (pairs/triples), samples
    sep_um: float             # min pairwise (x, y) separation
    resolvable: bool
    snippet: object


def _peak_snr(s, E, noise, lam=20.0):
    x, y, z, a, _ = s
    r = np.sqrt((E[:, 0] - x) ** 2 + (E[:, 1] - y) ** 2 + (E[:, 2] - z) ** 2)
    return a * float(np.max(lam / np.sqrt(r ** 2 + lam ** 2))) / noise


def _resolvable(src, E, noise) -> bool:
    if len(src) < 2:
        return False
    xy = np.array([s[:2] for s in src]); amps = [s[3] for s in src]
    sep = min(float(np.linalg.norm(xy[i] - xy[j])) for i in range(len(src)) for j in range(i + 1, len(src)))
    snr = min(_peak_snr(s, E, noise) for s in src)
    return bool(sep >= 60 and snr >= 6 and min(amps) / max(amps) >= 1 / 3)


def make_stress(pool: str, n_per_cell: int = 20, array=None):
    """Families: 'dt' pairs (sep 60-120 um) over dt bins; 'sep' pairs over separation bins (dt 1-9); triples;
    singles. Noise 5/10/20 uV, amplitudes 60-160 uV (as Phase 9)."""
    array = array or reference_array()
    E = array.xyz_um()
    rng = np.random.default_rng(SEEDS[pool])
    ylo, yhi = E[:, 1].min() + 40, E[:, 1].max() - 40
    out = {"dt": [], "sep": [], "triple": [], "single": []}

    def src0():
        return (float(rng.uniform(0, 70)), float(rng.uniform(ylo, yhi)), float(rng.uniform(0, 30)), float(rng.uniform(60, 160)), 60)

    def other(s, sep_lo, sep_hi, dt):
        dy = float(rng.choice([-1, 1]) * rng.uniform(sep_lo, sep_hi))
        return (float(rng.uniform(0, 70)), float(np.clip(s[1] + dy, E[:, 1].min() + 20, E[:, 1].max() - 20)),
                float(rng.uniform(0, 30)), float(s[3] * rng.uniform(0.5, 1.0)), 60 + int(rng.choice([-1, 1])) * dt)

    def emit(fam, kind, src, noise):
        tr = _render(src, E, noise, rng, 20.0, 15.0, 0.5)
        sn = extract_snippet(tr, 60, array, event_id=len(out[fam]), sample_rate_hz=FS)
        xy = np.array([s[:2] for s in src])
        sep = min((float(np.linalg.norm(xy[i] - xy[j])) for i in range(len(src)) for j in range(i + 1, len(src))), default=0.0)
        dt = abs(src[1][4] - src[0][4]) if len(src) > 1 else 0
        out[fam].append(StressEvent(kind, noise, tuple(src), dt, sep, _resolvable(src, E, noise), sn))

    for noise in (5.0, 10.0, 20.0):
        for lo, hi in ((0, 0), (1, 3), (4, 9), (10, 20)):
            for _ in range(n_per_cell):
                s = src0(); emit("dt", "pair", [s, other(s, 60, 120, int(rng.integers(lo, hi + 1)))], noise)
        for lo, hi in ((20, 40), (40, 60), (60, 90), (90, 140)):
            for _ in range(n_per_cell):
                s = src0(); emit("sep", "pair", [s, other(s, lo, hi, int(rng.integers(1, 10)))], noise)
        for _ in range(2 * n_per_cell):
            s = src0()
            b = other(s, 60, 120, int(rng.integers(1, 10)))
            # third source on the far side of the first (keeps pairwise separation >= 60 um where geometry allows)
            side = -np.sign(b[1] - s[1]) or 1.0
            c = (float(rng.uniform(0, 70)), float(np.clip(s[1] + side * rng.uniform(60, 120), E[:, 1].min() + 20, E[:, 1].max() - 20)),
                 float(rng.uniform(0, 30)), float(s[3] * rng.uniform(0.5, 1.0)), 60 + int(rng.choice([-1, 1]) * rng.integers(1, 10)))
            emit("triple", "triple", [s, b, c], noise)
        for _ in range(2 * n_per_cell):
            emit("single", "single", [src0()], noise)
    return array, out


def match(truth_xy: np.ndarray, est_xy: np.ndarray) -> dict:
    if len(est_xy) == 0:
        return {"matched": 0, "errs": [], "extra": 0, "missed": len(truth_xy)}
    D = np.linalg.norm(truth_xy[:, None] - est_xy[None], axis=2)
    ri, ci = linear_sum_assignment(D)
    ok = D[ri, ci] <= GATE_UM
    m = int(ok.sum())
    return {"matched": m, "errs": D[ri, ci][ok].tolist(), "extra": int(len(est_xy) - m), "missed": int(len(truth_xy) - m)}


# --- greedy K=3 extension (experiment) -------------------------------------------------------------------------
def k3_fit(loc: Phase8Localizer, sn, pos2: np.ndarray):
    """From an accepted K=2 solution, add a third source at the best grid cell orthogonal to the span of the first
    two whitened gains and refine all three. Returns (pos3 [3,2], delta3 = sse2 - sse3) on the same whitening."""
    f = loc.fitter
    Wt, sigma = f.whiten(sn)
    sse2 = f._vp_sse(pos2, Wt, sigma)
    g = f._P(_gain(pos2, f.E, f.lam, f.model) / sigma[None, :])      # [2,C]
    Q, _ = np.linalg.qr(g.T)                                           # [C,2] orthonormal span
    Gw = f._P(f.G / sigma[None, :])
    R = Gw - (Gw @ Q) @ Q.T
    norm = np.linalg.norm(R, axis=1)
    ok = norm > f.min_ortho * np.linalg.norm(Gw, axis=1)
    e = np.sum(((R / np.maximum(norm, 1e-18)[:, None]) @ Wt) ** 2, axis=1); e[~ok] = -1
    p3 = np.vstack([pos2, f.grid[int(np.argmax(e))]])
    p3 = f._refine(p3, Wt, sigma)
    sse3 = f._vp_sse(p3, Wt, sigma)
    return p3, float(sse2 - sse3)


def detected_dead_mask(sn) -> np.ndarray | None:
    """Usable-contact mask from the data alone (flat/near-silent contacts, same rule as the P9 conditioner)."""
    dead = dead_channels(np.asarray(sn.voltages_uv, float), sn.n_pre)
    return None if not dead.any() else ~dead


def evaluate_event(loc, ev, mask=None, zero=None, k3_tau=None, dead_as_mask=False) -> dict:
    """``dead_as_mask``: Phase 10 experiment X1 - contacts detected as dead are excluded from the fit (mask) instead
    of being interpolated (Phase 9 behaviour)."""
    sn = ev.snippet
    if zero is not None:
        v = np.asarray(sn.voltages_uv, float).copy(); v[:, zero] = 0.0
        sn = type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um,
                      sn.peak_channel, sn.ground_truth_xyz_um)
    if mask is not None:
        v = np.asarray(sn.voltages_uv, float).copy(); v[:, ~mask] = 0.0
        sn = type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um,
                      sn.peak_channel, sn.ground_truth_xyz_um)
    if dead_as_mask:
        det = detected_dead_mask(sn)
        if det is not None:
            mask = det if mask is None else (mask & det)
        c = condition(sn, baseline=True)
    else:
        c = condition(sn, **OV_COND)
    r = loc.localize(c, mask=mask)
    est = np.array([s.xyz_um[:2] for s in r.sources]) if r.sources else np.zeros((0, 2))
    n_est = r.n_sources
    d3 = None
    if k3_tau is not None and r.accepted and n_est == 2 and "shared_waveform_pair" not in r.flags:
        p3, d3 = k3_fit(loc, c, est)
        if d3 > k3_tau and min(np.linalg.norm(p3[2] - p3[i]) for i in range(2)) >= loc.cal.min_sep_um:
            est, n_est = p3, 3
    truth = np.array([s[:2] for s in ev.sources])
    m = match(truth, est)
    primary = float(np.linalg.norm(est[0] - truth[0])) if len(est) else np.nan
    return {"kind": ev.kind, "noise": ev.noise_uv, "dt": ev.dt, "sep": ev.sep_um, "res": ev.resolvable,
            "n_est": n_est, "accepted": r.accepted, "flags": r.flags, "delta3": d3, "primary_err": primary,
            "ev_err_nearest": float(np.mean(np.linalg.norm(truth[:, None] - est[None], axis=2).min(axis=1))) if len(est) else np.nan,
            **m, "ms": r.runtime_ms}


def summarize(rows, n_true: int) -> dict:
    if not rows:
        return {"n": 0}
    full = [r for r in rows if r["accepted"] and r["n_est"] >= n_true and r["matched"] == n_true]
    errs = [e for r in rows if r["accepted"] for e in r["errs"]]
    return {"n": len(rows), "accepted": float(np.mean([r["accepted"] for r in rows])),
            "declared": {str(k): float(np.mean([r["n_est"] == k for r in rows])) for k in (1, 2, 3)},
            "all_sources_recovered": float(len(full) / len(rows)),
            "mean_sources_matched": float(np.mean([r["matched"] for r in rows])),
            "phantom_rate": float(np.mean([r["extra"] > 0 for r in rows])),
            "flagged_possible_overlap": float(np.mean(["possible_unresolved_overlap" in r["flags"] for r in rows])),
            "matched_err_median_um": float(np.median(errs)) if errs else None,
            "matched_err_p90_um": float(np.percentile(errs, 90)) if errs else None,
            "runtime_ms_median": float(np.median([r["ms"] for r in rows]))}
