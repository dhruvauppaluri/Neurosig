"""Phase 14.7: fast K <= 3 path for the frozen P14 count model (ms3 basis, spatial whitening).

Same statistic as edge/phase14/neuron_model.SourceModel, restructured so that everything that depends only on the
contact geometry, the live-contact mask and the session's spatial whitening is computed once and cached:
  coarse grid (10 um) and fine grid (2.5 um) whitened, common-mode-projected bases B [G, C, m], their orthonormal
  bases Q and Gram matrices B^T B.
Per event only matrix products remain: K=1 energies Q^T D (one GEMM), and for each candidate first centre the
energy of every second centre by the Gram-matrix projection identity
  E([Q1, B2]) = |Q1^T D|^2 + tr(x^T (B2^T B2 - C^T C)^{-1} x),  C = Q1^T B2,  x = B2^T D - C^T Q1^T D
(batched 3 x 3 solves). Local refinement uses the precomputed fine grid (+-10 um neighbourhoods). Every event gets
the full K=1 / K=2 / (if two) K=3 analysis: no event is skipped for speed. Equivalence with SourceModel is a
regression test (edge/tests/test_phase14.py).
"""

from __future__ import annotations

import numpy as np

from edge.phase14.neuron_model import SPECS, Prepared, kernel_cols


def _whiten(K, pr: Prepared):
    B = np.einsum("dc,gcm->gdm", pr.Ws, K[:, pr.live]) if pr.Ws is not None else K[:, pr.live] / pr.sigma[pr.live][None, :, None]
    return B - pr.u[None, :, None] * np.einsum("c,gcm->gm", pr.u, B)[:, None, :]


class _Grid:
    def __init__(self, P, K, pr: Prepared):
        B = _whiten(K, pr)
        self.P, self.B = P, B
        q, r = np.linalg.qr(B)
        self.Q = q * (np.abs(np.diagonal(r, axis1=-2, axis2=-1)) > 1e-9)[..., None, :]
        self.G2 = np.einsum("gcm,gcn->gmn", B, B)
        G, C, m = B.shape
        self.Bf = B.transpose(0, 2, 1).reshape(G * m, C)       # rows = (g, k)
        self.Qf = self.Q.transpose(0, 2, 1).reshape(G * m, C)
        self.m = m


class _LazyGrid:
    """Fine grid whose whitened bases are computed only for the grid points an event actually visits (a new
    dead-contact mask then costs a few hundred points, not the full 2.5 um grid)."""

    def __init__(self, P, K, pr: Prepared):
        self.P, self.K, self.pr = P, K, pr
        G, _, m = K.shape; C = int(pr.live.sum()); self.m = m
        self.B = np.zeros((G, C, m)); self.Q = np.zeros((G, C, m)); self.G2 = np.zeros((G, m, m)); self.done = np.zeros(G, bool)

    def ensure(self, idx):
        idx = np.asarray(idx, int)
        new = idx[~self.done[idx]]
        if len(new):
            B = _whiten(self.K[new], self.pr)
            q, r = np.linalg.qr(B)
            self.B[new] = B; self.Q[new] = q * (np.abs(np.diagonal(r, axis1=-2, axis2=-1)) > 1e-9)[..., None, :]
            self.G2[new] = np.einsum("gcm,gcn->gmn", B, B); self.done[new] = True


class FastSourceModel:
    def __init__(self, E, spec: str = "ms3", coarse: float = 10.0, fine: float = 2.5, x_range=(-10.0, 90.0),
                 y_margin: float = 40.0, min_sep: float = 20.0, n_cand: int = 4, rounds: int = 2):
        self.E = np.asarray(E, float)[:, :2]; self.spec = SPECS[spec]; self.m = len(self.spec); self.min_sep = min_sep
        self.n_cand, self.rounds = n_cand, rounds
        ylo, yhi = self.E[:, 1].min() - y_margin, self.E[:, 1].max() + y_margin

        def grid(step):
            xs = np.arange(x_range[0], x_range[1] + 1e-9, step); ys = np.arange(ylo, yhi + 1e-9, step)
            return np.array([(x, y) for y in ys for x in xs]), len(xs), len(ys)
        self.Pc, self.nxc, self.nyc = grid(coarse); self.Pf, self.nxf, self.nyf = grid(fine)
        self.Kc = kernel_cols(self.Pc, self.E, self.spec); self.Kf = kernel_cols(self.Pf, self.E, self.spec)
        self.fine = fine; self.lo = np.array([x_range[0], ylo]); self.hi = np.array([x_range[1], yhi])
        self._cache = {}

    def _grids(self, pr: Prepared):
        key = (pr.live.tobytes(), id(pr.Ws) if pr.Ws is not None else pr.sigma.tobytes())
        g = self._cache.get(key)
        if g is None:
            if len(self._cache) > 64:
                self._cache.clear()
            g = (_Grid(self.Pc, self.Kc, pr), _LazyGrid(self.Pf, self.Kf, pr))
            self._cache[key] = g
        return g

    # ------------------------------------------------------------------ primitives
    @staticmethod
    def _e1(grid: _Grid, D, idx=None):
        Qf = grid.Qf if idx is None else grid.Q[idx].transpose(0, 2, 1).reshape(-1, D.shape[0])
        X = (Qf @ D).reshape(-1, grid.m, D.shape[1])
        return np.sum(X * X, axis=(1, 2))

    @staticmethod
    def _orth_set(Bs):
        q, r = np.linalg.qr(np.concatenate(Bs, axis=1))
        return q * (np.abs(np.diag(r)) > 1e-9)[None]

    @staticmethod
    def _inv(M):
        if M.shape[1] != 3:
            return np.linalg.inv(M)
        a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
        d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
        g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
        A = e * i - f * h; Bc = -(d * i - f * g); Cc = d * h - e * g
        det = a * A + b * Bc + c * Cc
        inv = np.stack([np.stack([A, -(b * i - c * h), b * f - c * e], 1),
                        np.stack([Bc, a * i - c * g, -(a * f - c * d)], 1),
                        np.stack([Cc, -(a * h - b * g), a * e - b * d], 1)], 1)
        return inv / np.where(np.abs(det) > 1e-300, det, 1e-300)[:, None, None]

    def _e_add(self, grid: _Grid, Q1, D, Q1D, BtD, idx=None, P0=None):
        """Energy of [Q1, B_g] for grid points idx (all if None), via E = |Q1^T D|^2 + tr(M^-1 S),
        M = B^T B - C^T C, S = x x^T, x = B^T D - C^T Q1^T D (all [m x m] per grid point)."""
        B = grid.B if idx is None else grid.B[idx]
        G2 = grid.G2 if idx is None else grid.G2[idx]
        C = np.matmul(Q1.T, B)                                                # [G, k, m]
        Ct = C.transpose(0, 2, 1)
        if P0 is None:
            P0 = np.matmul(BtD, BtD.transpose(0, 2, 1))                       # [G, m, m]
        Z = np.matmul(BtD, Q1D.T)                                             # [G, m, k]
        YY = Q1D @ Q1D.T                                                      # [k, k]
        ZC = np.matmul(Z, C)
        S = P0 - ZC - ZC.transpose(0, 2, 1) + np.matmul(np.matmul(Ct, YY), C)
        M = G2 - np.matmul(Ct, C) + 1e-9 * np.eye(B.shape[2])[None]
        return float(np.trace(YY)) + np.sum(self._inv(M) * S.transpose(0, 2, 1), axis=(1, 2))

    def _neigh(self, p, span=10.0):
        ix = np.flatnonzero((np.abs(self.Pf[:, 0] - p[0]) <= span + 1e-9) & (np.abs(self.Pf[:, 1] - p[1]) <= span + 1e-9))
        return ix

    # ------------------------------------------------------------------ fit
    def fit(self, pr: Prepared, kmax: int = 3, f21_thr: float | None = None, grid_map: bool = False) -> dict:
        gc, gf = self._grids(pr)
        D = pr.D; T = D.shape[1]; tot = float(np.sum(D * D)); n = pr.n_obs; m = self.m
        e1c = self._e1(gc, D)
        i1 = int(np.argmax(e1c))
        nb = self._neigh(self.Pc[i1]); gf.ensure(nb); e1f = self._e1(gf, D, nb); j = int(np.argmax(e1f))
        p1i = int(nb[j]); en1 = float(e1f[j])
        out = {"tot": tot, "sse1": tot - en1, "p1": self.Pf[p1i].copy(), "T": T}
        if grid_map:
            out["sse1_grid"] = tot - e1c
        if kmax < 2:
            return out
        BtD_c = (gc.Bf @ D).reshape(-1, m, T)
        P0_c = np.matmul(BtD_c, BtD_c.transpose(0, 2, 1))
        order = np.argsort(-e1c); cands = []
        for jj in order:
            if all(np.linalg.norm(self.Pc[jj] - self.Pc[c]) >= 20 for c in cands):
                cands.append(int(jj))
            if len(cands) == self.n_cand:
                break
        best = None
        for jj in cands:
            Q1 = gc.Q[jj]; Q1D = Q1.T @ D
            e = self._e_add(gc, Q1, D, Q1D, BtD_c, P0=P0_c)
            far = np.linalg.norm(self.Pc - self.Pc[jj], axis=1) >= self.min_sep
            e = np.where(far, e, -np.inf); k = int(np.argmax(e))
            if best is None or e[k] > best[0]:
                best = (float(e[k]), self.Pc[jj].copy(), self.Pc[k].copy())
        _, a, b = best
        en2 = best[0]
        for _ in range(self.rounds):
            for which in (0, 1):
                fixed, mov = (b, a) if which == 0 else (a, b)
                fi = int(np.argmin(np.linalg.norm(self.Pf - fixed, axis=1)))
                nb = self._neigh(mov)
                nb = nb[np.linalg.norm(self.Pf[nb] - fixed, axis=1) >= self.min_sep]
                if not len(nb):
                    continue
                gf.ensure(np.r_[nb, fi])
                Q1 = gf.Q[fi]; Q1D = Q1.T @ D
                BtD = np.matmul(gf.B[nb].transpose(0, 2, 1), D)
                e = self._e_add(gf, Q1, D, Q1D, BtD, nb); k = int(np.argmax(e))
                if e[k] >= en2 - 1e-9:
                    en2 = float(e[k])
                    if which == 0:
                        a = self.Pf[nb[k]].copy()
                    else:
                        b = self.Pf[nb[k]].copy()
        out.update(sse2=tot - en2, p2=np.stack([a, b]))
        out["F21"] = ((out["sse1"] - out["sse2"]) / (m * T)) / max(out["sse2"] / max(n - 2 * m * T, 1), 1e-12)
        out["chi1"] = out["sse1"] / max(n - m * T, 1)
        if kmax >= 3 and (f21_thr is None or out["F21"] > f21_thr):
            ia = int(np.argmin(np.linalg.norm(self.Pf - a, axis=1))); ib = int(np.argmin(np.linalg.norm(self.Pf - b, axis=1)))
            gf.ensure([ia, ib])
            Q12 = self._orth_set([gf.B[ia], gf.B[ib]]); Q12D = Q12.T @ D
            e = self._e_add(gc, Q12, D, Q12D, BtD_c, P0=P0_c)
            far = (np.linalg.norm(self.Pc - a, axis=1) >= self.min_sep) & (np.linalg.norm(self.Pc - b, axis=1) >= self.min_sep)
            e = np.where(far, e, -np.inf); k = int(np.argmax(e))
            out.update(sse3=tot - float(e[k]), p3=np.stack([a, b, self.Pc[k]]))
            out["F32"] = ((out["sse2"] - out["sse3"]) / (m * T)) / max(out["sse3"] / max(n - 3 * m * T, 1), 1e-12)
        return out
