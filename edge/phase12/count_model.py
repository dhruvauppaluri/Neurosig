"""Phase 12.4: offline analysis records and source-count decisions.

``analyze`` runs the frozen Phase 9/10 K<=2 space-time fitter once per event (dead contacts masked, baseline
conditioning, as P11) and keeps everything a decision rule needs: K=1, free K=2 and shared-waveform K=2 positions,
SSE drops, chi2_red, Gauss-Newton covariances, the search box and the temporal features. Decisions are then applied
offline, so every rule is scored on identical fits:

  P11       OV-A rule with the frozen Phase 9 thresholds (Phase 11 final configuration)
  P12-LR    logistic model on the six count features (FEATURES), trained on p12dev only; threshold set so that the
            p12dev single-source false-two rate is 2% (leaves room for a <= 5% upper confidence bound elsewhere)
  gate      cheap pre-screen (rank2, n_trough, t_spread, chi1) deciding whether the K=2 fit is run at all
The merge (< min_sep) and axial boundary checks of P11 apply to every rule unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from edge.phase8.multisource import _gain
from edge.phase9.improved import condition
from edge.phase10 import overlap_stress as O
from edge.phase12.overlap_bench import FEATURES, feature_vector, temporal_features

DOF_FREE = 6.0      # extra parameters of the free K=2 model (2 positions + rank-4 waveform)
DOF_SHARED = 3.0    # extra parameters of the shared-waveform K=2 model


def analyze(loc, sn) -> dict:
    mask = O.detected_dead_mask(sn)
    c = condition(sn, baseline=True)
    f = loc.fitter
    t = time.perf_counter()
    Wt, sigma = f.whiten(c, mask)
    t_k1 = time.perf_counter()
    fit = f.fit(c, mask=mask)
    ms_full = (time.perf_counter() - t) * 1e3
    if fit is None:
        return {"none": True, "ms_full": ms_full}
    # K=1 only cost (grid + refine), measured separately for the gated runtime
    t1 = time.perf_counter()
    e1 = f._grid_energy(Wt, sigma, None); p1 = f.grid[int(np.argmax(e1))].copy(); f._refine(p1[None, :], Wt, sigma)
    ms_k1 = (time.perf_counter() - t1) * 1e3 + (t_k1 - t) * 1e3
    tt = time.perf_counter()
    tf = temporal_features(c, sigma, mask)
    ms_feat = (time.perf_counter() - tt) * 1e3
    pos = lambda S: np.array([[s.x, s.y] for s in S])
    chi1 = fit.sse1 / max(Wt.size - 2 - Wt.shape[1], 1)
    c1 = max(chi1, 1.0)
    out = {"none": False, "pos1": pos(fit.sources1), "pos2": pos(fit.sources2), "pos_sh": pos(fit.sources_shared),
           "en2": [s.energy for s in fit.sources2], "en_sh": [s.energy for s in fit.sources_shared],
           "delta": fit.delta, "dshared": fit.delta_shared, "chi1": chi1, "delta_norm": fit.delta / c1,
           "delta_shared_norm": fit.delta_shared / c1, "sse1": fit.sse1, "sse2": fit.sse2, "n_obs": Wt.size,
           "rank": Wt.shape[1], "lo": f.lo.copy(), "hi": f.hi.copy(), "n_dead": 0 if mask is None else int((~mask).sum()),
           "ms_full": ms_full, "ms_k1": ms_k1, "ms_feat": ms_feat, **tf}
    for key in ("pos1", "pos2", "pos_sh"):
        try:
            out["cov_" + key] = f.covariance(out[key], Wt, sigma)
        except Exception:  # noqa: BLE001
            out["cov_" + key] = np.full((2 * len(out[key]), 2 * len(out[key])), np.nan)
    v = np.asarray(c.voltages_uv, float); amp = -v[c.n_pre]
    nz = np.median(np.abs(v[:20] - np.median(v[:20], axis=0)), axis=0) / 0.6745
    live = np.ones(v.shape[1], bool) if mask is None else mask
    out["snr_obs"] = float(np.max(amp[live] / np.maximum(nz[live], 1e-6)))
    out.update(second_source_features(f, c, sigma, mask, out["pos2"]))
    out["x"] = feature_vector(out, tf)
    return out


def second_source_features(f, sn, sigma, mask, pos2) -> dict:
    """Waveforms of the free K=2 solution on the full-rank whitened window [-0.3, +0.6] ms: polarity of each source
    (trough depth / max |h|), correlation of the two waveforms, energy ratio and in-plane separation. A dendritic or
    return-current component of one extended neuron tends to be weak, positive-dominant and adjacent; a second neuron
    has its own negative trough."""
    v = np.asarray(sn.voltages_uv, float)
    live = np.ones(v.shape[1], bool) if mask is None else np.asarray(mask, bool)
    k = sn.n_pre
    W = (v[max(k - 9, 0):k + 18] / np.where(live, sigma, np.inf)[None]).T          # [C, T]
    g = _gain(pos2, f.E, f.lam, f.model) / np.where(live, sigma, np.inf)[None]
    u = live / np.where(live, sigma, np.inf); u = u / max(np.linalg.norm(u), 1e-12)
    W = W - np.outer(u, u @ W); g = g - (g @ u)[:, None] * u[None]
    h, *_ = np.linalg.lstsq(g.T, W, rcond=None)                                     # [2, T]
    en = np.sum((g[:, :, None] * h[:, None, :]) ** 2, axis=(1, 2))
    o = np.argsort(-en)
    h, en = h[o], en[o]
    pol = [float(-x.min() / max(np.abs(x).max(), 1e-12)) for x in h]
    cc = float(np.corrcoef(h[0], h[1])[0, 1]) if np.std(h[1]) > 0 and np.std(h[0]) > 0 else 0.0
    return {"pol1": pol[0], "pol2": pol[1], "hcorr": cc, "en_ratio": float(en[1] / max(en[0], 1e-12)),
            "sep2": float(np.linalg.norm(pos2[0] - pos2[1]))}


@dataclass
class Decision:
    n: int
    pos: np.ndarray
    cov: np.ndarray
    flags: list
    kind: str            # one | free | shared


def finish(a: dict, two: bool, use_free: bool, cal) -> Decision:
    flags = []
    if not two:
        return Decision(1, a["pos1"], a["cov_pos1"], flags, "one")
    pos = a["pos2"] if use_free else a["pos_sh"]
    cov = a["cov_pos2"] if use_free else a["cov_pos_sh"]
    if np.linalg.norm(pos[0] - pos[1]) < cal.min_sep_um:
        return Decision(1, a["pos1"], a["cov_pos1"], ["merged_close_pair"], "one")
    lo, hi = a["lo"], a["hi"]
    if np.any(pos[:, 1] >= hi[1] - 1.0) or np.any(pos[:, 1] <= lo[1] + 1.0):
        return Decision(1, a["pos1"], a["cov_pos1"], ["boundary_pair_rejected"], "one")
    if np.any(pos[:, 0] >= hi[0] - 1.0):
        flags.append("lateral_limit_source")
    if not use_free:
        flags.append("shared_waveform_pair")
    en = a["en2"] if use_free else a["en_sh"]
    order = np.argsort([-e for e in en])
    pos = pos[order]
    idx = np.concatenate([[2 * k, 2 * k + 1] for k in order])
    return Decision(2, pos, None if cov is None else cov[np.ix_(idx, idx)], flags, "free" if use_free else "shared")


def decide_p11(a: dict, cal) -> Decision:
    free = a["delta"] > cal.tau_two
    shared = (not free) and a["dshared"] > cal.tau_shared
    d = finish(a, free or shared, free, cal)
    if d.n == 1 and not d.flags and a["delta"] > cal.tau_gray:
        d.flags.append("possible_unresolved_overlap")
    return d


class LogisticCount:
    """Standardized logistic regression (L2, C = 1) on FEATURES; fitted on p12dev records only."""

    def __init__(self):
        self.mu = self.sd = self.w = None
        self.thr = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray, singles_X: np.ndarray, target_fp: float = 0.02):
        from scipy.optimize import minimize
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd]); y = np.asarray(y, float)

        def nll(w):          # L2 (C = 1) on the slopes, not the intercept
            z = Z @ w
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(w[1:] ** 2)), \
                Z.T @ (1.0 / (1.0 + np.exp(-z)) - y) + np.concatenate([[0.0], w[1:]])
        self.w = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        self.thr = float(np.quantile(self.prob(singles_X), 1 - target_fp))
        return self

    def prob(self, X):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(self.w[0] + Z @ self.w[1:])))

    def to_json(self):
        return {"features": list(FEATURES), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), "thr": self.thr}

    @classmethod
    def from_json(cls, d):
        m = cls(); m.mu, m.sd, m.w, m.thr = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"]), float(d["thr"]); return m


def decide_lr(a: dict, cal, model: LogisticCount) -> Decision:
    two = bool(model.prob(a["x"])[0] > model.thr)
    use_free = (a["delta_norm"] / DOF_FREE) >= (a["delta_shared_norm"] / DOF_SHARED)
    return finish(a, two, use_free, cal)


class Gate:
    """Invoke the K=2 fit iff any cheap statistic exceeds its threshold (thresholds from p12dev)."""

    KEYS = ("rank2", "n_trough", "t_spread", "chi1")

    def __init__(self, thr: dict | None = None):
        self.thr = thr or {}

    def invoke(self, a: dict) -> bool:
        return any(a[k] > self.thr[k] for k in self.KEYS)

    def calibrate(self, pos_records: list[dict], neg_records: list[dict], keep: float = 0.99, grid=40):
        """Choose a common quantile q on the negatives (singles): thr_k = q-quantile of k over singles; smallest
        invoke-rate q grid point such that >= keep of the positives (pairs the full rule recovers) still invoke."""
        best = None
        for q in np.linspace(0.5, 0.99, grid):
            thr = {k: float(np.quantile([r[k] for r in neg_records], q)) for k in self.KEYS}
            self.thr = thr
            rec = np.mean([self.invoke(r) for r in pos_records])
            inv = np.mean([self.invoke(r) for r in neg_records])
            if rec >= keep and (best is None or inv < best[0]):
                best = (inv, dict(thr))
        self.thr = best[1] if best else {k: -np.inf for k in self.KEYS}
        return self


def score(ev, d: Decision, gate_um: float = 60.0) -> dict:
    m = O.match(ev.sources, d.pos)
    primary = float(np.linalg.norm(d.pos[0] - ev.sources[0])) if len(d.pos) else np.nan
    # nearest-truth error of the primary (identity-agnostic) and its own assignment
    near = float(np.min(np.linalg.norm(ev.sources - d.pos[0], axis=1)))
    return {"n_est": d.n, **m, "primary_err": primary, "primary_nearest_err": near, "flags": d.flags, "kind_dec": d.kind}


def analyze_fast(loc, sn) -> dict | None:
    """Production path: only what the P12-LR decision needs (no covariances, no duplicate K=1 timing)."""
    mask = O.detected_dead_mask(sn)
    c = condition(sn, baseline=True)
    f = loc.fitter
    Wt, sigma = f.whiten(c, mask)
    fit = f.fit(c, mask=mask)
    if fit is None:
        return None
    tf = temporal_features(c, sigma, mask)
    pos = lambda S: np.array([[s.x, s.y] for s in S])
    chi1 = fit.sse1 / max(Wt.size - 2 - Wt.shape[1], 1); c1 = max(chi1, 1.0)
    out = {"pos1": pos(fit.sources1), "pos2": pos(fit.sources2), "pos_sh": pos(fit.sources_shared),
           "en2": [s.energy for s in fit.sources2], "en_sh": [s.energy for s in fit.sources_shared],
           "delta": fit.delta, "dshared": fit.delta_shared, "chi1": chi1, "delta_norm": fit.delta / c1,
           "delta_shared_norm": fit.delta_shared / c1, "lo": f.lo, "hi": f.hi, **tf}
    out.update(second_source_features(f, c, sigma, mask, out["pos2"]))
    out["x"] = feature_vector(out, tf)
    for key in ("pos1", "pos2", "pos_sh"):
        out["cov_" + key] = None
    return out
