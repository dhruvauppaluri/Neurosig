"""Phase 11G staged edge pipeline.

Stage 1  fast single-source estimate on every event: P10 weighted_scan (frozen P9 engine, dead contacts excluded).
Stage 2  ambiguity score (cheap, cached vectorized grid on the conditioned trough-amplitude vector):
         G1 "maxres"  max |residual|/sigma after the best single line source (rejected: invokes ~94% of events)
         G2 "greedy2" F-like ratio of the best orthogonal second-source energy drop to the remaining residual.
Stage 3  only if score > tau_gate: the K<=2 space-time pipeline with the 11E count rule (OV-A).
Stage 4  output: estimate(s), Mondrian-calibrated region (11F), flags, accepted/rejected with reason.
tau_gate is set on p11dev so that >= 98% of the events that the full pipeline declares two-source pass the gate.
"""

from __future__ import annotations

import time

import numpy as np

from edge.phase9.improved import condition, dead_channels
from edge.phase10.configs import p10_factory
from edge.phase10.overlap_stress import localizer
from edge.phase11.overlap import CountRule, localize as count_localize

RULE = CountRule(axial_only_boundary=True)


class AmbiguityScore:
    def __init__(self, array, lam=20.0, step=8.0, mode="greedy2"):
        self.mode = mode
        E = array.xyz_um()
        gx = np.arange(-20.0, 80.1, step); gy = np.arange(E[:, 1].min() - 40, E[:, 1].max() + 40.1, step)
        P = np.array(np.meshgrid(gx, gy, indexing="ij")).reshape(2, -1).T
        r2 = (P[:, None, 0] - E[None, :, 0]) ** 2 + (P[:, None, 1] - E[None, :, 1]) ** 2
        self.G = lam / np.sqrt(r2 + lam * lam)

    def __call__(self, sn, live=None):
        v = np.asarray(sn.voltages_uv, float)
        k = max(5, sn.n_pre - 12)
        v = v - np.median(v[:k], axis=0, keepdims=True); v = v - np.median(v, axis=1, keepdims=True)
        a = -v[sn.n_pre]
        sig = np.maximum(np.median(np.abs(v[:k] - np.median(v[:k], axis=0)), axis=0) / 0.6745, 1e-6)
        w = 1.0 / sig ** 2
        if live is not None:
            w = np.where(live, w, 0.0)
        num = self.G @ (a * w); den = (self.G * self.G) @ w
        amp = np.maximum(num / np.maximum(den, 1e-18), 0.0)
        sse = float(np.sum(a * a * w)) - 2 * amp * num + amp * amp * den
        i = int(np.argmin(sse))
        rss1 = float(sse[i])
        if self.mode == "maxres":
            res = (a - amp[i] * self.G[i]) / sig
            if live is not None:
                res = np.where(live, res, 0.0)
            return float(np.max(np.abs(res))), rss1 / max(np.sum(a * a * w), 1e-12)
        # "greedy2": best second line source orthogonal to the first (whitened), F-like ratio
        sw = np.sqrt(w)
        g1 = self.G[i] * sw; u1 = g1 / max(np.linalg.norm(g1), 1e-18)
        r = a * sw - (a * sw @ u1) * u1
        Gw = self.G * sw[None]; Gw = Gw - (Gw @ u1)[:, None] * u1[None]
        nrm = np.linalg.norm(Gw, axis=1); ok = nrm > 0.25 * np.linalg.norm(self.G * sw[None], axis=1)
        drop = np.where(ok, (Gw @ r) ** 2 / np.maximum(nrm, 1e-18) ** 2, 0.0)
        d2 = float(drop.max()); n = int((w > 0).sum())
        F = (d2 / 2.0) / max((rss1 - d2) / max(n - 4, 1), 1e-12)
        return float(F), rss1 / max(np.sum(a * a * w), 1e-12)


class StagedLocalizer:
    def __init__(self, array, tau_gate: float, cal=None, rule: CountRule = RULE, mode="greedy2"):
        self.array = array
        self.fast = p10_factory(array)["weighted_scan"]
        self.score = AmbiguityScore(array, mode=mode)
        self.loc = localizer(array, cal)
        self.tau = tau_gate
        self.rule = rule

    def localize(self, sn):
        t0 = time.perf_counter()
        live = ~dead_channels(np.asarray(sn.voltages_uv, float), sn.n_pre)
        fast = self.fast.estimate(sn)
        s, rel = self.score(sn, live)
        out = {"fast": fast, "score": s, "rel_residual": rel, "refined": None, "invoked": s > self.tau}
        if out["invoked"]:
            c = condition(sn, baseline=True)
            mask = live if (~live).any() else None
            out["refined"], _ = count_localize(self.loc, c, self.rule, mask)
        out["ms"] = (time.perf_counter() - t0) * 1e3
        return out
