"""Phase 14 frozen event pipeline (P14): source count, positions, calibrated overlap probability, uncertainty.

Per session (label-free): a NoiseModel of the event-free background (spatial covariance) and the self-calibrated
count thresholds (F21 at the 98th percentile and K=3 F32 rule) from development-morphology singles placed in that
background. Per event:
  count      FastSourceModel(ms3, spatial whitening) K<=2 statistic F21; 'two' iff F21 > thr2; 'three' iff two and
             F32 > thr3 (thr3 = 95th percentile of F32 over development pairs, i.e. <= 5% false three on pairs)
  p_overlap  calibrated probability of >= 2 sources: logistic map of log F21 (fit on development singles + pairs,
             balanced) converted to the stated deployment prior (PRIOR_OVERLAP) by Bayes' rule
  position   K=1: P12 single-source path (E3 alignment, mean-3 trough amplitudes, FastLine20) or the ms3 centre
             (chosen on dev; see frozen_p14.json); K=2/3: the ms3 centres
  region     primary-source profile-likelihood region on the 10 um ms3 grid, {g : SSE1_g - SSE1_min <= q chi1},
             q conformally calibrated on the complete development population for 90% primary coverage; a region
             touching the search-box border is reported as unbounded (never clipped to look precise)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from edge.phase14.fast14 import FastSourceModel
from edge.phase14.neuron_model import NoiseModel, prepare

PRIOR_OVERLAP = 0.10


@dataclass
class P14Config:
    spec: str = "ms3"
    n_cand: int = 2
    rounds: int = 1
    fp_target: float = 0.02
    q_region: float = 4.0
    k1_position: str = "fastline20"
    lr_w: tuple = (0.0, 1.0)          # logistic on log F21: logit = w0 + w1 * log F21 (balanced fit)
    thr3: float = np.inf


class SessionContext:
    """Per-session state: noise model, thresholds, model cache."""

    def __init__(self, noise_segments: np.ndarray, cfg: P14Config):
        self.noise = NoiseModel(noise_segments)
        self.cfg = cfg
        self.thr2 = np.inf
        self._models = {}

    def model(self, E):
        k = np.asarray(E, float).tobytes()
        if k not in self._models:
            self._models[k] = FastSourceModel(E, self.cfg.spec, n_cand=self.cfg.n_cand, rounds=self.cfg.rounds)
        return self._models[k]

    def calibrate(self, cal_snippets) -> float:
        F = [self.model(np.asarray(s.electrode_xyz_um)[:, :2]).fit(prepare(s, noise=self.noise, temporal=False), 2)["F21"]
             for s in cal_snippets]
        self.thr2 = float(np.quantile(F, 1 - self.cfg.fp_target))
        return self.thr2


def p_overlap(F21: float, cfg: P14Config, prior: float = PRIOR_OVERLAP) -> float:
    z = cfg.lr_w[0] + cfg.lr_w[1] * np.log(max(F21, 1e-9))
    lr = np.exp(np.clip(z, -50, 50))                 # balanced fit: odds = likelihood ratio
    o = lr * prior / (1 - prior)
    return float(o / (1 + o))


def run_event(ctx: SessionContext, sn, with_region: bool = True) -> dict:
    cfg = ctx.cfg
    t = time.perf_counter()
    E = np.asarray(sn.electrode_xyz_um, float)[:, :2]
    pr = prepare(sn, noise=ctx.noise, temporal=False)
    m = ctx.model(E)
    f = m.fit(pr, 3, f21_thr=ctx.thr2, grid_map=with_region)
    two = f["F21"] > ctx.thr2
    three = bool(two and f.get("F32", -np.inf) > cfg.thr3)
    n = 3 if three else (2 if two else 1)
    out = {"n": n, "F21": float(f["F21"]), "F32": float(f.get("F32", np.nan)), "p_overlap": p_overlap(f["F21"], cfg),
           "p1_ms3": f["p1"], "p2": f.get("p2"), "p3": f.get("p3"), "chi1": f["chi1"]}
    out["ms_count"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    if cfg.k1_position == "fastline20":
        from edge.phase12.fast import cheap_features, FastLine20
        v = np.asarray(sn.voltages_uv, float)
        key = E.tobytes() + b"fl"
        if key not in ctx._models:
            ctx._models[key] = FastLine20(np.c_[E, np.zeros(len(E))])
        fl = ctx._models[key]
        from edge.phase8.refine import channel_noise
        th, area, _ = cheap_features(v, sn.n_pre, fl, channel_noise(v))
        out["p1"] = np.asarray(th, float) if th is not None else f["p1"]; out["p12_area"] = area
    else:
        out["p1"] = f["p1"]
    out["ms_position"] = (time.perf_counter() - t) * 1e3
    if with_region:
        sse = f["sse1_grid"]; c = max(f["chi1"], 1.0)
        out["region_delta"] = (sse - f["sse1"]) / c                  # per coarse grid point, for offline calibration
    return out


def region(delta: np.ndarray, q: float, P: np.ndarray, lo: np.ndarray, hi: np.ndarray, step: float = 10.0):
    """(mask, area um^2, unbounded) of the profile-likelihood region at level q."""
    m = delta <= q
    if not m.any():
        m = delta <= delta.min()
    border = (np.abs(P[:, 0] - lo[0]) < 1e-6) | (np.abs(P[:, 0] - hi[0]) < 1e-6) | (np.abs(P[:, 1] - lo[1]) < 1e-6) | (np.abs(P[:, 1] - hi[1]) < 1e-6)
    return m, float(m.sum() * step * step), bool(np.any(m & border))


def covered(delta, q, P, truth, step: float = 10.0) -> bool:
    """Truth inside the region: the coarse cell containing the truth (nearest grid point) is in the region."""
    i = int(np.argmin(np.sum((P - truth[None]) ** 2, axis=1)))
    inside_box = np.all(np.abs(P[i] - truth) <= step / 2 + 1e-6)
    return bool(inside_box and delta[i] <= q)
