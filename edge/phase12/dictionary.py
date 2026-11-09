"""Phase 12.4c: empirical spatiotemporal waveform dictionary as the single-neuron null model for source counting.

Motivation (12.4 dev): a line source with a free temporal waveform cannot represent one extended biophysical neuron,
so a second 'source' absorbs the dendritic / return-current component. P11 declares two sources for 67% of
noise-free MEArec singles and for 85% of real matched events. Here one source is one dictionary atom: a complete
multichannel waveform of a single neuron, translated along the probe, mirrored laterally, with a free non-negative
amplitude and a small time shift. Two sources are two atoms with independent amplitudes and timing.

Atoms live on the Neuropixels 1.0 checkerboard lattice (x in {11, 27, 43, 59} um, 20 um rows, alternating row types
{11, 43} / {27, 59}); a placement keeps the row type (40 um steps) and the lateral mirror x -> 70 - x flips it, so
together atoms can be placed every 20 um. Data and atoms are both median-CAR'd and noise-weighted per contact.
Count statistic: F = (R1 - R2) / (R2 / dof), dof = n_obs - (number of fitted parameters); threshold calibrated on
development singles only. Positions: the line20 fit of each demixed component (data minus the other atom).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from edge.phase11 import forward_models as F

PRE, POST = 20, 40                 # atom window around its trough (samples at 30 kHz)
TAU1 = 3                           # time-shift search of the first atom
TAU2 = 22                          # relative time-shift search of the second atom
MIN_SEP2 = 40.0                    # second atom must sit >= 40 um from the first (a resolvable pair is >= 60 um)
X_MIRROR = 70.0


@dataclass
class Atom:
    table: dict                    # (x, drow) -> waveform [PRE + POST]
    peak_type: frozenset           # x-pair of the peak contact's row
    src_rel: np.ndarray            # source (x, y - y_peak_row) if known, else peak contact position
    name: str


def row_type(E: np.ndarray, y: float) -> frozenset:
    return frozenset(np.round(E[np.abs(E[:, 1] - y) < 0.5, 0]).astype(int).tolist())


def make_atom(W: np.ndarray, E: np.ndarray, t0: int, src_xy: np.ndarray | None, name: str) -> Atom | None:
    if t0 - PRE < 0 or t0 + POST > W.shape[0]:
        return None
    w = W[t0 - PRE:t0 + POST]
    w = w - np.median(w, axis=1, keepdims=True)
    pk = int(np.argmin(w[PRE]))
    ypk = E[pk, 1]
    nrm = max(-w[PRE, pk], 1e-9)
    table = {(int(round(E[c, 0])), int(round((E[c, 1] - ypk) / 20.0))): w[:, c] / nrm for c in range(W.shape[1])}
    rel = (np.array([src_xy[0], src_xy[1] - ypk]) if src_xy is not None else np.array([E[pk, 0], 0.0]))
    return Atom(table, row_type(E, ypk), rel, name)


def mirror(a: Atom) -> Atom:
    return Atom({(int(X_MIRROR - x), r): v for (x, r), v in a.table.items()},
                frozenset(int(X_MIRROR - x) for x in a.peak_type), np.array([X_MIRROR - a.src_rel[0], a.src_rel[1]]),
                a.name + "_m")


class Dictionary:
    def __init__(self, atoms: list[Atom]):
        self.atoms = atoms + [mirror(a) for a in atoms]
        self._geo = {}

    def candidates(self, E: np.ndarray, exclude: frozenset = frozenset()):
        """All placements on contact set E: G [n, T, C] (median-CAR'd), source positions [n, 2]. Atoms whose base
        name is in ``exclude`` are skipped (leave-template-out evaluation)."""
        key = (E.tobytes(), exclude)
        if key in self._geo:
            return self._geo[key]
        rows = np.unique(E[:, 1])
        G, P = [], []
        for a in self.atoms:
            if a.name.removesuffix("_m") in exclude:
                continue
            for y in rows:
                if row_type(E, y) != a.peak_type:
                    continue
                g = np.zeros((PRE + POST, len(E)))
                for c in range(len(E)):
                    v = a.table.get((int(round(E[c, 0])), int(round((E[c, 1] - y) / 20.0))))
                    if v is not None:
                        g[:, c] = v
                g = g - np.median(g, axis=1, keepdims=True)
                if np.abs(g).max() < 1e-9:
                    continue
                G.append(g); P.append([a.src_rel[0], a.src_rel[1] + y])
        out = (np.array(G), np.array(P))
        self._geo[key] = out
        return out

    def fit(self, v: np.ndarray, k: int, sigma: np.ndarray, E: np.ndarray, mask: np.ndarray | None = None,
            exclude: frozenset = frozenset()) -> dict:
        """``v`` raw snippet [T, C], reference sample k. Returns K=1 / K=2 fits and the count statistic."""
        G, P = self.candidates(E, exclude)
        live = np.ones(v.shape[1], bool) if mask is None else np.asarray(mask, bool)
        w = np.where(live, 1.0 / sigma, 0.0)
        x = (v - np.median(v, axis=1, keepdims=True)) * w[None]
        pad = TAU2 + PRE + 5                      # zero padding so short snippets can host a shifted second atom
        x = np.pad(x, ((pad, pad), (0, 0))); k = k + pad
        Gw = G * w[None, None]
        n, L, C = Gw.shape
        Gf = Gw.reshape(n, -1); gg = np.sum(Gf * Gf, axis=1) + 1e-12
        # K = 1: first atom over small time shifts
        lo = k - PRE - TAU1
        X1 = sliding_window_view(x[lo:lo + L + 2 * TAU1], (L, C))[:, 0].reshape(2 * TAU1 + 1, -1)     # [S, L*C]
        num = X1 @ Gf.T                                                                              # [S, n]
        amp = np.maximum(num / gg[None], 0.0)
        sc = amp * num
        s1, j1 = np.unravel_index(int(np.argmax(sc)), sc.shape)
        tot1 = float(np.sum(X1[s1] ** 2))
        R1 = tot1 - float(sc[s1, j1])
        t1 = lo + s1                                   # start sample of the first atom
        # K = 2: top first candidates, best second atom over a wide relative shift on a padded residual
        best = (R1, None)
        firsts = np.argsort(-sc[s1])[:5]
        lo2 = t1 - TAU2
        if lo2 >= 0 and t1 + L + TAU2 <= x.shape[0]:
            Xw = x[lo2:t1 + L + TAU2]
            for j in firsts:
                if amp[s1, j] <= 0:
                    continue
                R = Xw.copy(); R[TAU2:TAU2 + L] -= amp[s1, j] * Gw[j]
                S = sliding_window_view(R, (L, C))[:, 0].reshape(2 * TAU2 + 1, -1)
                num2 = S @ Gf.T
                far = np.linalg.norm(P - P[j], axis=1) >= MIN_SEP2
                a2 = np.maximum(num2 / gg[None], 0.0); sc2 = np.where(far[None], a2 * num2, -1.0)
                s2, j2 = np.unravel_index(int(np.argmax(sc2)), sc2.shape)
                if sc2[s2, j2] <= 0:
                    continue
                # joint non-negative LS of both amplitudes on the union window
                A = np.zeros((2, Xw.size))
                g1 = np.zeros_like(Xw); g1[TAU2:TAU2 + L] = Gw[j]
                g2 = np.zeros_like(Xw); g2[s2:s2 + L] = Gw[j2]
                A[0], A[1] = g1.ravel(), g2.ravel()
                coef, *_ = np.linalg.lstsq(A.T, Xw.ravel(), rcond=None)
                coef = np.maximum(coef, 0.0)
                Rj = float(np.sum((Xw.ravel() - A.T @ coef) ** 2))
                R1w = float(np.sum((Xw.ravel() - A[0] * max(float(A[0] @ Xw.ravel()) / max(A[0] @ A[0], 1e-12), 0.0)) ** 2))
                # compare on the same (padded) window: R1 recomputed there
                if best[1] is None or (R1w - Rj) > best[0]:
                    best = (R1w - Rj, (j, j2, s2 - TAU2, coef, R1w, Rj, lo2, Xw.size))
        out = {"R1": R1, "j1": int(j1), "t1": int(t1 - pad), "amp1": float(amp[s1, j1]), "pos1_dict": P[j1].copy(), "n_obs1": int(live.sum() * L)}
        if best[1] is None:
            out.update(F=0.0, two=None)
            return out
        j, j2, dt2, coef, R1w, Rj, lo2, nobs = best[1]
        dof = max(int(live.sum()) * (L + 2 * TAU2) - 6, 1)
        out.update(F=float((R1w - Rj) / max(Rj / dof, 1e-12)), two={"j1": int(j), "j2": int(j2), "dt2": int(dt2),
                   "coef": coef.tolist(), "pos_dict": np.array([P[j], P[j2]]), "t_first": int(t1 - pad), "lo2": int(lo2 - pad)})
        out["_G"] = (G, P)
        return out


def demixed_positions(fit: dict, v: np.ndarray, sigma: np.ndarray, E: np.ndarray, cache: dict) -> np.ndarray:
    """line20 position of each source after subtracting the other source's fitted atom (raw units, CAR'd)."""
    G, P = fit["_G"]
    tw = fit["two"]
    x = v - np.median(v, axis=1, keepdims=True)
    L = PRE + POST
    t1 = tw["t_first"]; t2 = t1 + tw["dt2"]
    comp = []
    for keep, other, t_k, t_o, c_o in ((tw["j1"], tw["j2"], t1, t2, tw["coef"][1]), (tw["j2"], tw["j1"], t2, t1, tw["coef"][0])):
        r = x.copy()
        a, b = max(t_o, 0), min(t_o + L, r.shape[0])
        r[a:b] -= c_o * G[other][a - t_o:b - t_o] * 1.0
        k = t_k + PRE
        if k - 40 < 0 or k >= r.shape[0]:
            comp.append(P[keep]); continue
        base = np.median(r[max(k - 40, 0):max(k - 12, 1)], axis=0)
        amp = -(r[k] - base)
        f = F.fit(F.Model("line20"), amp, sigma, E, cache=cache)
        comp.append(f.theta[:2] if f.amps[0] > 0 else P[keep])
    return np.array(comp)
