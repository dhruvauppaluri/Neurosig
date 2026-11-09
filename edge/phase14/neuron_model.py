"""Phase 14.2-14.3: single-neuron spatiotemporal null models and source-count statistics.

A neuron's extracellular waveform is not a separable (one spatial map x one temporal waveform) point/line source:
somatic, dendritic and return currents have different spatial spreads and timing. If the one-neuron model cannot
express that, a two-source fit absorbs the misfit and a phantom second neuron is declared (Phase 12: P11 did so for
67-85% of MEArec singles). Phase 14 makes the null richer instead of the alternative.

Model family: a source at in-plane centre p = (lateral, axial) spans a small spatial basis B(p) [C, m]; each basis
column carries its own free temporal waveform (rank m at one centre):
  line20   g_20(p)                                    m = 1   (the Phase 9-12 space-time model)
  ms       g_15, g_40, g_100 at p and d g_40 / d axial  m = 4   multi-scale + axial dipole ('extended neuron')
  ms5      ms + d g_40 / d lateral                     m = 5
with the line-source kernel g_lam(p, e) = lam / sqrt(|p - e|^2 + lam^2). K sources are K centres with concatenated
bases. Data: baseline-conditioned snippet, per-contact noise-whitened, common mode projected out, window -24..+36
samples around the reference trough, dead contacts masked. Fits: exhaustive coarse grid (10 um), best K=1 centre,
K=2 by orthogonal projection over all second centres for the top-4 first centres, then alternating 2 um local
refinement; K=3 greedily adds a third centre to the best pair.

Count statistic: F21 = [(SSE1 - SSE2) / (m T)] / [SSE2 / (n_obs - 2 m T)] (and F32 likewise). Under model mismatch
F is not F-distributed; thresholds are calibrated on development-morphology singles only.
Component descriptors for the K=2 solution (whitened units): each component's reconstruction B_k H_k, its trough
depth, polarity (-min / max|.|), trough time and energy share; a second neuron has its own negative trough, a
dendritic / return-current component of one neuron tends to be weak and positive or adjacent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge.phase9.improved import condition, dead_channels, edge_noise

WIN = (-24, 36)
SPECS = {
    "line20": (("g", 20.0),),
    "ms": (("g", 15.0), ("g", 40.0), ("g", 100.0), ("dy", 40.0)),
    "ms5": (("g", 15.0), ("g", 40.0), ("g", 100.0), ("dy", 40.0), ("dx", 40.0)),
    "msq": (("g", 15.0), ("g", 40.0), ("g", 100.0), ("dy", 40.0), ("dyy", 40.0)),     # + axial quadrupole
    "msn": (("g", 10.0), ("g", 30.0), ("g", 90.0), ("dy", 30.0)),                     # narrower scales
    "msw": (("g", 20.0), ("g", 60.0), ("g", 150.0), ("dy", 60.0)),                    # wider scales
    "ms3": (("g", 15.0), ("g", 40.0), ("dy", 40.0)),                                  # m = 3
}


def kernel_cols(P: np.ndarray, E: np.ndarray, spec) -> np.ndarray:
    """[n_p, C, m] basis columns at centres P [n_p, 2] for contacts E [C, 2]."""
    dx = P[:, None, 0] - E[None, :, 0]; dy = P[:, None, 1] - E[None, :, 1]
    cols = []
    for kind, lam in spec:
        q = dx * dx + dy * dy + lam * lam
        if kind == "g":
            cols.append(lam / np.sqrt(q))
        elif kind == "dy":
            cols.append(-lam * dy * q ** -1.5 * lam)          # scaled by lam so columns have comparable size
        elif kind == "dx":
            cols.append(-lam * dx * q ** -1.5 * lam)
        elif kind == "dyy":
            cols.append(lam ** 3 * (3 * dy * dy * q ** -2.5 - q ** -1.5))
    return np.stack(cols, axis=2)


@dataclass
class Prepared:
    D: np.ndarray            # [C_live, T] whitened, common mode removed
    live: np.ndarray         # bool [C]
    sigma: np.ndarray        # [C]
    u: np.ndarray            # [C_live] unit common-mode vector
    E: np.ndarray            # [C, 2]
    n_obs: int
    Ws: np.ndarray | None = None      # [C_live, C_live] spatial whitening (None = diagonal 1 / sigma)
    Wt: np.ndarray | None = None      # [T, T] temporal whitening applied on the right (D = Ws X Wt^T)
    Ws_inv: np.ndarray | None = None
    Wt_inv: np.ndarray | None = None


class NoiseModel:
    """Separable spatio-temporal noise covariance of the conditioned background (Sigma_s x Sigma_t).

    Estimated from event-free raw segments [n, T_seg, C] processed exactly like an event snippet (baseline median of
    the first n_pre - 12 samples, window WIN around a nominal reference). In deployment this is a running estimate
    from the recording's own event-free samples; no event or label is used. Shrinkage: 1% of the mean variance."""

    def __init__(self, segs: np.ndarray, n_pre: int = 40, offset: int = 10, win=WIN, shrink: float = 0.01):
        X = np.asarray(segs, float)[:, offset:offset + 100]
        k0 = max(5, n_pre - 12)
        X = X - np.median(X[:, :k0], axis=1, keepdims=True)
        X = X[:, n_pre + win[0]:n_pre + win[1]]                    # [n, T, C]
        self.T = X.shape[1]
        S = np.einsum("ntc,ntd->cd", X, X) / (X.shape[0] * X.shape[1])
        self.Ss = S + shrink * np.trace(S) / len(S) * np.eye(len(S))
        self.sigma = np.sqrt(np.diag(self.Ss))
        L = np.linalg.cholesky(self.Ss)
        Z = np.einsum("cd,ntd->ntc", np.linalg.inv(L), X)
        St = np.einsum("ntc,nsc->ts", Z, Z) / (Z.shape[0] * Z.shape[2])
        self.St = St + shrink * np.trace(St) / len(St) * np.eye(len(St))
        Lt = np.linalg.cholesky(self.St)
        self.Wt = np.linalg.inv(Lt); self.Wt_inv = Lt
        self._ws = {}

    def spatial(self, live: np.ndarray):
        key = live.tobytes()
        if key not in self._ws:
            S = self.Ss[np.ix_(live, live)]
            L = np.linalg.cholesky(S)
            self._ws[key] = (np.linalg.inv(L), L)
        return self._ws[key]


def prepare(sn, win=WIN, noise: NoiseModel | None = None, temporal: bool = True) -> Prepared:
    c = condition(sn, baseline=True)
    v = np.asarray(c.voltages_uv, float)
    k = sn.n_pre
    dead = dead_channels(np.asarray(sn.voltages_uv, float), k)
    live = ~dead
    E = np.asarray(sn.electrode_xyz_um, float)[:, :2]
    seg = v[max(k + win[0], 0):min(k + win[1], v.shape[0])][:, live]
    if noise is None:
        sigma = edge_noise(v, k)
        sigma = np.maximum(sigma, 0.2 * np.median(sigma[live]))
        u = np.ones(int(live.sum())) / sigma[live]; u /= np.linalg.norm(u)
        D = (seg / sigma[live][None]).T
        D = D - np.outer(u, u @ D)
        return Prepared(D, live, sigma, u, E, int(D.size))
    Ws, Ws_inv = noise.spatial(live)
    Wt, Wt_inv = (noise.Wt, noise.Wt_inv) if (temporal and seg.shape[0] == noise.T) else (np.eye(seg.shape[0]), np.eye(seg.shape[0]))
    D = Ws @ seg.T @ Wt.T
    u = Ws @ np.ones(int(live.sum())); u /= np.linalg.norm(u)
    D = D - np.outer(u, u @ D)
    return Prepared(D, live, noise.sigma.copy(), u, E, int(D.size), Ws, Wt, Ws_inv, Wt_inv)


class SourceModel:
    """Grid-based K <= 3 fits for one basis family on one contact geometry."""

    def __init__(self, E: np.ndarray, spec: str = "ms", step: float = 10.0, x_range=(-10.0, 90.0), y_margin=40.0,
                 min_sep: float = 20.0):
        self.E = np.asarray(E, float)[:, :2]
        self.spec = SPECS[spec]; self.name = spec; self.m = len(self.spec)
        xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
        ys = np.arange(self.E[:, 1].min() - y_margin, self.E[:, 1].max() + y_margin + 1e-9, step)
        self.P = np.array([(x, y) for y in ys for x in xs])
        self.Kc = kernel_cols(self.P, self.E, self.spec)             # [G, C, m] unwhitened
        self.lo = np.array([x_range[0], ys[0]]); self.hi = np.array([x_range[1], ys[-1]])
        self.min_sep = min_sep; self.step = step

    # ---------------------------------------------------------------- helpers
    def _basis(self, pr: Prepared, P: np.ndarray | None = None) -> np.ndarray:
        K = self.Kc if P is None else kernel_cols(P, self.E, self.spec)
        if pr.Ws is not None:
            B = np.einsum("dc,gcm->gdm", pr.Ws, K[:, pr.live])
        else:
            B = K[:, pr.live] / pr.sigma[pr.live][None, :, None]
        return B - pr.u[None, :, None] * np.einsum("c,gcm->gm", pr.u, B)[:, None, :]

    @staticmethod
    def _orth(B: np.ndarray) -> np.ndarray:
        q, r = np.linalg.qr(B)
        ok = np.abs(np.diagonal(r, axis1=-2, axis2=-1)) > 1e-9
        return q * ok[..., None, :]

    def _energy(self, Q: np.ndarray, D: np.ndarray) -> np.ndarray:
        X = np.einsum("gcm,ct->gmt", Q, D)
        return np.sum(X * X, axis=(1, 2))

    def _pair_energy(self, Q1: np.ndarray, B2: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Energy captured by [Q1, B2_g] for every g (Q1 fixed, orthonormal)."""
        coef = np.einsum("cm,gcn->gmn", Q1, B2)                        # [G, m1, m2]
        R = B2 - np.einsum("cm,gmn->gcn", Q1, coef)
        Q2 = self._orth(R)
        Dres = D - Q1 @ (Q1.T @ D)
        X = np.einsum("gcm,ct->gmt", Q2, Dres)
        return float(np.sum((Q1.T @ D) ** 2)) + np.sum(X * X, axis=(1, 2))

    def _local(self, pr, centre, others_Q, D, span=10.0, step=2.0):
        """Best centre in a local 2 um grid around ``centre`` given the other centres' orthonormal basis."""
        g = np.arange(-span, span + 1e-9, step)
        P = np.array([(centre[0] + a, centre[1] + b) for b in g for a in g])
        P = np.clip(P, self.lo, self.hi)
        B = self._basis(pr, P)
        if others_Q is None:
            e = self._energy(self._orth(B), D)
        else:
            e = self._pair_energy(others_Q, B, D)
        i = int(np.argmax(e))
        return P[i], float(e[i])

    def _Q_of(self, pr, centres) -> np.ndarray:
        B = self._basis(pr, np.atleast_2d(centres))                      # [k, C, m]
        return self._orth(np.concatenate(list(B), axis=1)[None])[0]

    # ---------------------------------------------------------------- fits
    def fit(self, pr: Prepared, kmax: int = 2) -> dict:
        D = pr.D; tot = float(np.sum(D * D)); T = D.shape[1]
        B = self._basis(pr)
        Q = self._orth(B)
        e1 = self._energy(Q, D)
        i1 = int(np.argmax(e1))
        p1, en1 = self._local(pr, self.P[i1], None, D)
        out = {"tot": tot, "sse1": tot - en1, "p1": p1, "T": T}
        if kmax < 2:
            return out
        # candidate first centres: top local maxima of e1 (20 um suppression)
        order = np.argsort(-e1); cands = []
        for j in order:
            if all(np.linalg.norm(self.P[j] - self.P[c]) >= 20 for c in cands):
                cands.append(j)
            if len(cands) == 4:
                break
        best = None
        for j in cands:
            Q1 = self._orth(B[j][None])[0]
            e = self._pair_energy(Q1, B, D)
            far = np.linalg.norm(self.P - self.P[j], axis=1) >= self.min_sep
            e = np.where(far, e, -np.inf)
            k = int(np.argmax(e))
            if best is None or e[k] > best[0]:
                best = (float(e[k]), self.P[j].copy(), self.P[k].copy())
        _, a, b = best
        for _ in range(2):
            a, _ = self._local(pr, a, self._Q_of(pr, b), D)
            b, en2 = self._local(pr, b, self._Q_of(pr, a), D)
        if np.linalg.norm(a - b) < self.min_sep:            # local refinement may not merge the pair
            en2 = float(np.sum((self._Q_of(pr, np.stack([a, b])).T @ D) ** 2))
        out.update(sse2=tot - en2, p2=np.stack([a, b]))
        out.update(self._components(pr, np.stack([a, b])))
        if kmax >= 3:
            Q12 = self._Q_of(pr, np.stack([a, b]))
            e = self._pair_energy(Q12, B, D)
            far = (np.linalg.norm(self.P - a, axis=1) >= self.min_sep) & (np.linalg.norm(self.P - b, axis=1) >= self.min_sep)
            e = np.where(far, e, -np.inf); k = int(np.argmax(e))
            c3, en3 = self._local(pr, self.P[k], Q12, D)
            out.update(sse3=tot - en3, p3=np.stack([a, b, c3]))
        n = pr.n_obs; m = self.m
        out["F21"] = ((out["sse1"] - out["sse2"]) / (m * T)) / max(out["sse2"] / max(n - 2 * m * T, 1), 1e-12)
        if "sse3" in out:
            out["F32"] = ((out["sse2"] - out["sse3"]) / (m * T)) / max(out["sse3"] / max(n - 3 * m * T, 1), 1e-12)
        out["chi1"] = out["sse1"] / max(n - m * T, 1)
        return out

    def _components(self, pr: Prepared, centres: np.ndarray) -> dict:
        B = self._basis(pr, centres)                                    # [2, C, m]
        A = np.concatenate(list(B), axis=1)
        H, *_ = np.linalg.lstsq(A, pr.D, rcond=None)
        comps = [B[k] @ H[k * self.m:(k + 1) * self.m] for k in range(len(centres))]   # [C, T] each (whitened)
        if pr.Ws is not None:                       # back to per-contact noise units (unwhitened shape, sigma-scaled)
            s = pr.sigma[pr.live][:, None]
            comps = [(pr.Ws_inv @ c @ pr.Wt_inv.T) / s for c in comps]
        en = np.array([float(np.sum(c * c)) for c in comps])
        o = np.argsort(-en)
        d = {}
        for r, k in enumerate(o):
            c = comps[k]
            d[f"c{r}_trough"] = float(-c.min()); d[f"c{r}_pol"] = float(-c.min() / max(np.abs(c).max(), 1e-12))
            d[f"c{r}_t"] = int(np.unravel_index(int(np.argmin(c)), c.shape)[1]); d[f"c{r}_en"] = float(en[k])
        d["order"] = o.tolist()
        d["en_ratio"] = float(en[o[1]] / max(en[o[0]], 1e-12))
        d["dt_comp"] = abs(d["c1_t"] - d["c0_t"])
        d["sep"] = float(np.linalg.norm(centres[0] - centres[1]))
        return d
