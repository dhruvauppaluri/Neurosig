"""Phase 15 Track F: batched deployment path (latency / throughput configuration D-P15).

Events that share a contact geometry and live-contact mask (the common case within one recording) are processed
together so that every stage is a few large matrix products instead of many small per-event calls:
  1. prepare: baseline + spatial whitening (session NoiseModel) + common-mode projection, per event (cheap), then
     temporal SVD compression of each whitened snippet to rank r (default 6; separation unchanged, tested)
  2. K=1: energies of all coarse-grid centres for all events in ONE GEMM, for ms3 and line20; sub-grid position by
     separable parabolic interpolation of the energy map around the maximum
  3. K=2 (ms3): for each event its top-2 first centres; energy of every second centre via the Gram identity of
     edge/phase14/fast14.py, batched over (events x candidates); no local refinement (the count threshold is
     self-calibrated on the same statistic, so it absorbs the grid approximation)
Outputs per event: F21 (count statistic), K=1 positions (ms3, line20, their mean = A-P15 estimator), K=2 centres.
Every event is processed; batching only changes when results are emitted (bounded by the batch size / timeout).
"""

from __future__ import annotations

import numpy as np

from edge.phase14.neuron_model import SPECS, kernel_cols, prepare


class BatchModel:
    def __init__(self, E, step: float = 10.0, x_range=(-10.0, 90.0), y_margin: float = 40.0, min_sep: float = 20.0,
                 rank: int = 6, n_cand: int = 2):
        self.E = np.asarray(E, float)[:, :2]
        xs = np.arange(x_range[0], x_range[1] + 1e-9, step); ys = np.arange(self.E[:, 1].min() - y_margin, self.E[:, 1].max() + y_margin + 1e-9, step)
        self.nx, self.ny, self.step = len(xs), len(ys), step
        self.P = np.array([(x, y) for y in ys for x in xs])
        self.K = {s: kernel_cols(self.P, self.E, SPECS[s]) for s in ("ms3", "line20")}
        self.rank, self.n_cand, self.min_sep = rank, n_cand, min_sep
        self.far = np.linalg.norm(self.P[:, None] - self.P[None], axis=2) >= min_sep          # [G, G]
        self._c = {}

    def _bases(self, pr):
        key = (pr.live.tobytes(), id(pr.Ws))
        if key not in self._c:
            if len(self._c) > 32:
                self._c.clear()
            out = {}
            for s, K in self.K.items():
                B = np.einsum("dc,gcm->gdm", pr.Ws, K[:, pr.live])
                B = B - pr.u[None, :, None] * np.einsum("c,gcm->gm", pr.u, B)[:, None, :]
                q, r = np.linalg.qr(B)
                Q = q * (np.abs(np.diagonal(r, axis1=-2, axis2=-1)) > 1e-9)[..., None, :]
                G, C, m = B.shape
                out[s] = dict(B=B, Q=Q, G2=np.einsum("gcm,gcn->gmn", B, B), Qf=Q.transpose(0, 2, 1).reshape(G * m, C),
                              Bf=B.transpose(0, 2, 1).reshape(G * m, C), m=m)
            self._c[key] = out
        return self._c[key]

    def _subgrid(self, e1: np.ndarray) -> np.ndarray:
        """Parabolic sub-grid peak of a [G] energy map on the (ny, nx) grid."""
        i = int(np.argmax(e1)); iy, ix = divmod(i, self.nx)
        M = e1.reshape(self.ny, self.nx)
        def off(a, b, c):
            den = a - 2 * b + c
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0
        dx = off(M[iy, ix - 1], M[iy, ix], M[iy, ix + 1]) if 0 < ix < self.nx - 1 else 0.0
        dy = off(M[iy - 1, ix], M[iy, ix], M[iy + 1, ix]) if 0 < iy < self.ny - 1 else 0.0
        return self.P[i] + self.step * np.array([dx, dy])

    def run(self, snippets, noise) -> list[dict]:
        prs = [prepare(sn, noise=noise, temporal=False) for sn in snippets]
        groups = {}
        for i, pr in enumerate(prs):
            groups.setdefault(pr.live.tobytes(), []).append(i)
        out = [None] * len(prs)
        for idx in groups.values():
            self._run_group([prs[i] for i in idx], idx, out)
        return out

    def _run_group(self, prs, idx, out):
        bs = self._bases(prs[0]); r = self.rank
        D3 = np.stack([pr.D for pr in prs])                                   # [N, C, T0] (same mask => same shape)
        if r and D3.shape[2] > r:
            U, S, _ = np.linalg.svd(D3, full_matrices=False); D3 = U[:, :, :r] * S[:, None, :r]
        N, C, T = D3.shape
        Ds = list(D3); nobs = [D3[0].size] * N
        Dall = D3.transpose(1, 0, 2).reshape(C, N * T)                        # [C, N*T]
        res = {}
        for s in ("ms3", "line20"):
            b = bs[s]; m = b["m"]
            X = b["Qf"] @ Dall
            res[s] = np.square(X).reshape(-1, m, N, T).sum(axis=(1, 3))       # [G, N]
        b = bs["ms3"]; m = b["m"]; G = len(self.P)
        BtD = (b["Bf"] @ Dall).reshape(G, m, N, T).transpose(2, 0, 1, 3)       # [N, G, m, T]
        P0 = np.matmul(BtD, BtD.transpose(0, 1, 3, 2))                         # [N, G, m, m]
        e1 = res["ms3"]
        from edge.phase14.fast14 import FastSourceModel
        # first-centre candidates per event (top n_cand, >= min_sep apart)
        cand = np.zeros((N, self.n_cand), int)
        for n in range(N):
            order = np.argsort(-e1[:, n]); c = [int(order[0])]
            for j in order[1:]:
                if len(c) == self.n_cand:
                    break
                if self.far[int(j), c].all():
                    c.append(int(j))
            while len(c) < self.n_cand:
                c.append(c[0])
            cand[n] = c
        J = cand.ravel(); nn = np.repeat(np.arange(N), self.n_cand)                 # [NC]
        Q1 = b["Q"][J]                                                             # [NC, C, m]
        Dn = np.stack(Ds)                                                          # [N, C, T]
        Y = np.matmul(Q1.transpose(0, 2, 1), Dn[nn])                               # [NC, m, T]
        YY = np.matmul(Y, Y.transpose(0, 2, 1))                                    # [NC, m, m]
        Bt = b.setdefault("Bt", b["B"].transpose(1, 0, 2).reshape(C, G * m))       # [C, G*m]
        Cm = (Q1.transpose(0, 2, 1).reshape(-1, C) @ Bt).reshape(len(J), m, G, m).transpose(0, 2, 1, 3)   # [NC, G, m, m]
        Ct = Cm.transpose(0, 1, 3, 2)
        Z = np.matmul(BtD[nn], Y.transpose(0, 2, 1)[:, None])                      # [NC, G, m, m]
        ZC = np.matmul(Z, Cm)
        S = P0[nn] - ZC - ZC.transpose(0, 1, 3, 2) + np.matmul(np.matmul(Ct, YY[:, None]), Cm)
        Mx = b["G2"][None] - np.matmul(Ct, Cm) + 1e-9 * np.eye(m)
        Minv = FastSourceModel._inv(Mx.reshape(-1, m, m)).reshape(Mx.shape)
        e2 = np.trace(YY, axis1=1, axis2=2)[:, None] + np.sum(Minv * S.transpose(0, 1, 3, 2), axis=(2, 3))   # [NC, G]
        e2 = np.where(self.far[J], e2, -np.inf)
        kbest = np.argmax(e2, axis=1); vbest = e2[np.arange(len(J)), kbest]
        vb = vbest.reshape(N, self.n_cand); kb = kbest.reshape(N, self.n_cand)
        for n in range(N):
            c = int(np.argmax(vb[n])); best = float(vb[n, c]); pair = (int(cand[n, c]), int(kb[n, c]))
            tot_n = float(np.sum(Ds[n] * Ds[n]))
            sse1 = tot_n - float(e1[:, n].max()); sse2 = tot_n - best
            F21 = ((sse1 - sse2) / (m * T)) / max(sse2 / max(nobs[n] - 2 * m * T, 1), 1e-12)
            p_ms3 = self._subgrid(e1[:, n]); p_l20 = self._subgrid(res["line20"][:, n])
            out[idx[n]] = {"F21": float(F21), "p1_ms3": p_ms3, "p1_line20": p_l20, "p1": 0.5 * (p_ms3 + p_l20),
                           "p2": self.P[list(pair)], "chi1": sse1 / max(nobs[n] - m * T, 1), "snr_obs": float(np.max(-prs[n].D.min(axis=1)))}
