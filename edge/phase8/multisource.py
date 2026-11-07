"""Variable-projection space-time fit for K quasi-static sources with free temporal waveforms.

Model (whitened): W[c,t] = sum_k g_c(p_k)/sigma_c * h_k(t) + noise. Waveforms h_k are solved by
linear least squares for each candidate set of positions, so the kernel shape need not be known.
Positions are searched on a grid (greedy, K=1 then K=2) and refined with bounded LM.
Model order uses the whitened SSE drop, with the threshold calibrated on single-source dev events.
Same quasi-static monopole family as the existing scans; not a new beamformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from edge.forward import LAMBDA_UM
from edge.localize.common import finish_estimate, finish_unavailable
from edge.phase8.refine import _gain, channel_noise
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet


@dataclass
class SourceFit:
    x: float
    y: float
    energy: float          # whitened energy of this source's waveform (sum h^2 * |g|^2 projection)
    amp_uv: float          # peak |h * g| in uV on its strongest channel


@dataclass
class MultiFit:
    sources1: list[SourceFit]
    sources2: list[SourceFit]
    sse1: float
    sse2: float
    delta: float            # sse1 - sse2 (whitened chi-square units, free waveforms)
    n_obs: int
    sources_shared: list[SourceFit] = field(default_factory=list)   # K=2 with one shared waveform
    delta_shared: float = 0.0                                      # sse1 - sse_shared (null ~ chi2(3))
    flags: list = field(default_factory=list)


class SpaceTimeFitter:
    def __init__(self, array: ElectrodeArray, lambda_um: float = LAMBDA_UM, model: str = "offset_reciprocal",
                 grid_step_um: tuple[float, float] = (6.0, 10.0), max_x_um: float = 90.0, y_margin_um: float = 40.0,
                 min_candidate_ortho: float = 0.25, prescreen_free: float = 0.0,
                 prescreen_shared: float = 0.0, rank: int | None = None, nuisance_cm: bool = False) -> None:
        self.array = array
        self.lam = lambda_um
        self.model = model
        self.E = array.xyz_um()
        ys = self.E[:, 1]
        self.max_x = max_x_um
        xs = self.E[:, 0]
        planar = float(np.ptp(xs)) > 1e-6          # array has lateral span: sources may lie on either side
        x_lo = float(xs.min()) - max_x_um * 0.5 if planar else 1.0
        x_hi = float(xs.max()) + max_x_um * (0.5 if planar else 1.0)
        self.planar = planar
        gx = np.arange(x_lo, x_hi + 0.1, grid_step_um[0])
        gy = np.arange(ys.min() - 40.0, ys.max() + 40.1, grid_step_um[1])
        self.grid = np.array(np.meshgrid(gx, gy, indexing="ij")).reshape(2, -1).T
        self.G = _gain(self.grid, self.E, self.lam, model)
        self.min_ortho = min_candidate_ortho
        self.rank = rank  # reduced temporal rank for fitting (None = full)
        self.nuisance_cm = nuisance_cm   # project the common-mode direction out of data and gains (shared noise)
        self._u = None
        self.pre_free = prescreen_free      # skip K=2 LM when the greedy grid drop is below this (0 = never skip)
        self.pre_shared = prescreen_shared  # skip the shared-waveform refinement below this grid gain
        self.lo = np.array([x_lo - 0.5 if planar else 0.5, ys.min() - y_margin_um])
        self.hi = np.array([x_hi, ys.max() + y_margin_um])

    def _P(self, gw: np.ndarray) -> np.ndarray:
        """Remove the common-mode direction from whitened gains (last axis = channels)."""
        if self._u is None:
            return gw
        return gw - (gw @ self._u)[..., None] * self._u

    # ---------------------------------------------------------------------------------------
    def whiten(self, snippet: Snippet, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Whitened [C,T] data. ``mask`` (bool [C], True = usable) marks dropped contacts: their noise is set
        huge so they carry no weight in any fit instead of looking like perfectly quiet channels."""
        v = np.asarray(snippet.voltages_uv, float)
        sigma = channel_noise(v)
        if mask is not None:
            sigma = np.where(np.asarray(mask, bool), sigma, 1e6)
            v = np.where(np.asarray(mask, bool)[None, :], v, 0.0)
        Wt = (v / sigma[None, :]).T  # [C,T]
        if self.nuisance_cm:
            u = 1.0 / sigma
            self._u = u / np.linalg.norm(u)
            Wt = Wt - np.outer(self._u, self._u @ Wt)
        else:
            self._u = None
        if self.rank is not None and self.rank < Wt.shape[1]:
            u, sv, _ = np.linalg.svd(Wt, full_matrices=False)
            Wt = u[:, : self.rank] * sv[None, : self.rank]  # [C,r] temporal-subspace coordinates
        return Wt, sigma

    def _vp_sse(self, pos: np.ndarray, Wt: np.ndarray, sigma: np.ndarray) -> float:
        g = self._P(_gain(pos.reshape(-1, 2), self.E, self.lam, self.model) / sigma[None, :])  # [K,C]
        h, *_ = np.linalg.lstsq(g.T, Wt, rcond=None)
        r = Wt - g.T @ h
        return float(np.sum(r * r))

    def _vp_resid(self, flat: np.ndarray, Wt: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        g = self._P(_gain(flat.reshape(-1, 2), self.E, self.lam, self.model) / sigma[None, :])
        h, *_ = np.linalg.lstsq(g.T, Wt, rcond=None)
        return (Wt - g.T @ h).ravel()

    def _refine(self, pos: np.ndarray, Wt: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        k = pos.shape[0]
        lo = np.tile(self.lo, k)
        hi = np.tile(self.hi, k)
        p0 = np.clip(pos.ravel(), lo + 1e-6, hi - 1e-6)
        sol = least_squares(self._vp_resid, p0, args=(Wt, sigma), bounds=(lo, hi), method="trf",
                            x_scale=10.0, max_nfev=40)
        return sol.x.reshape(k, 2)

    def _grid_energy(self, Wt: np.ndarray, sigma: np.ndarray, exclude_unit: np.ndarray | None):
        Gw = self._P(self.G / sigma[None, :])
        if exclude_unit is not None:
            proj = Gw @ exclude_unit
            Gw = Gw - proj[:, None] * exclude_unit[None, :]
        norm = np.linalg.norm(Gw, axis=1)
        ok = norm > 1e-12
        if exclude_unit is not None:
            ok &= norm > self.min_ortho * np.linalg.norm(self.G / sigma[None, :], axis=1)
        unit = Gw / np.maximum(norm, 1e-18)[:, None]
        e = np.sum((unit @ Wt) ** 2, axis=1)
        e[~ok] = -1.0
        return e

    # -- shared-waveform (rank-1 temporal) two-source model -------------------------------------
    def _shared_lambda(self, g: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
        """Largest generalized eigenvalue of (G^T R G, G^T G) for g [2,C] whitened gains; returns (lam, coef)."""
        M = g.T
        A = M.T @ M + 1e-12 * np.eye(2)
        B = M.T @ R @ M
        w, v = np.linalg.eig(np.linalg.solve(A, B))
        # same-sign (physical) coefficients only: pick the best eigenpair with v1*v2 > 0
        best, coef = -1.0, np.ones(2)
        for i in range(2):
            vi = v[:, i].real
            if vi[0] * vi[1] > 0 and w.real[i] > best:
                best, coef = float(w.real[i]), vi
        return best, coef

    def _shared_sse(self, pos: np.ndarray, R: np.ndarray, total: float, sigma: np.ndarray) -> float:
        g = self._P(_gain(pos.reshape(-1, 2), self.E, self.lam, self.model) / sigma[None, :])
        lam, _ = self._shared_lambda(g, R)
        return total - lam if lam >= 0 else total

    def _fit_shared(self, pos1: np.ndarray, Wt: np.ndarray, sigma: np.ndarray, total: float):
        R = Wt @ Wt.T
        Gw = self._P(self.G / sigma[None, :])
        g1 = self._P(_gain(pos1, self.E, self.lam, self.model)[0] / sigma)
        # vectorized generalized eigenvalue over candidate second positions
        Rg1 = R @ g1
        a11 = g1 @ g1
        b11 = g1 @ Rg1
        a12 = Gw @ g1
        a22 = np.sum(Gw * Gw, axis=1)
        b12 = Gw @ Rg1
        b22 = np.sum((Gw @ R) * Gw, axis=1)
        det = a11 * a22 - a12 ** 2 + 1e-18
        # N = A^-1 B ; eigenvalues of 2x2
        n11 = (a22 * b11 - a12 * b12) / det
        n12 = (a22 * b12 - a12 * b22) / det
        n21 = (-a12 * b11 + a11 * b12) / det
        n22 = (-a12 * b12 + a11 * b22) / det
        tr = n11 + n22
        dsc = np.sqrt(np.maximum((n11 - n22) ** 2 / 4 + n12 * n21, 0.0))
        lam_hi = tr / 2 + dsc
        lam_lo = tr / 2 - dsc
        # eigenvector of N for eigenvalue l is (n12, l - n11); require same sign components
        ok_hi = n12 * (lam_hi - n11) > 0
        ok_lo = n12 * (lam_lo - n11) > 0
        lam = np.where(ok_hi, lam_hi, np.where(ok_lo, lam_lo, -1.0))
        ortho = np.sqrt(np.maximum(det, 0)) / np.maximum(np.sqrt(a11 * a22), 1e-18)
        lam[ortho < 0.05] = -1.0
        i2 = int(np.argmax(lam))
        init = np.concatenate([pos1[0], self.grid[i2]])
        base_lam = float(g1 @ Rg1 / max(a11, 1e-18))
        sh_grid = float(max(lam[i2] - base_lam, 0.0))
        if sh_grid < self.pre_shared:
            return init.reshape(2, 2), total - float(lam[i2]) if lam[i2] >= 0 else total, sh_grid
        res = minimize(lambda q: self._shared_sse(q, R, total, sigma) + 1e3 * (np.any(q.reshape(2, 2) < self.lo) or
                       np.any(q.reshape(2, 2) > self.hi)), init, method="Powell",
                       options={"maxiter": 200, "xtol": 0.3, "ftol": 1e-3})
        pos = np.clip(res.x.reshape(2, 2), self.lo, self.hi)
        sse = self._shared_sse(pos, R, total, sigma)
        return pos, sse, sh_grid

    def fit(self, snippet: Snippet, refine: bool = True, mask: np.ndarray | None = None) -> "MultiFit | None":
        Wt, sigma = self.whiten(snippet, mask)
        total = float(np.sum(Wt * Wt))
        if total <= 0:
            return None
        e1 = self._grid_energy(Wt, sigma, None)
        i1 = int(np.argmax(e1))
        p1 = self.grid[i1].copy()
        pos1 = self._refine(p1[None, :], Wt, sigma) if refine else p1[None, :]
        sse1 = self._vp_sse(pos1, Wt, sigma)
        g1 = self._P(_gain(pos1, self.E, self.lam, self.model)[0] / sigma)
        u1 = g1 / np.linalg.norm(g1)
        e2 = self._grid_energy(Wt, sigma, u1)
        i2 = int(np.argmax(e2))
        pos2_init = np.vstack([pos1[0], self.grid[i2]])
        grid_drop = float(max(e2[i2], 0.0))
        if refine and grid_drop >= self.pre_free:
            pos2 = self._refine(pos2_init, Wt, sigma)
            sse2 = self._vp_sse(pos2, Wt, sigma)
        else:
            pos2 = pos2_init
            sse2 = sse1 - grid_drop
        if sse2 > sse1:  # refinement should never be worse than K=1 plus a nuisance source
            sse2 = sse1
            pos2 = pos2_init
        pos_sh, sse_sh, sh_grid = self._fit_shared(pos1, Wt, sigma, total)
        sse_sh = min(sse_sh, sse1)
        shared = self._describe(pos_sh, Wt, sigma)
        return MultiFit(self._describe(pos1, Wt, sigma), self._describe(pos2, Wt, sigma), sse1, sse2,
                        sse1 - sse2, Wt.size, shared, sse1 - sse_sh)

    def covariance(self, pos: np.ndarray, Wt: np.ndarray, sigma: np.ndarray, step_um: float = 0.5) -> np.ndarray:
        """Gauss-Newton covariance of the K*2 position parameters (VP Jacobian), scaled by max(chi2_red, 1)."""
        flat = pos.ravel()
        r0 = self._vp_resid(flat, Wt, sigma)
        J = np.empty((r0.size, flat.size))
        for j in range(flat.size):
            d = np.zeros_like(flat)
            d[j] = step_um
            J[:, j] = (self._vp_resid(flat + d, Wt, sigma) - r0) / step_um
        dof = max(r0.size - flat.size - pos.shape[0] * Wt.shape[1], 1)
        scale = max(float(r0 @ r0) / dof, 1.0)
        return np.linalg.pinv(J.T @ J) * scale

    def _describe(self, pos: np.ndarray, Wt, sigma) -> list[SourceFit]:
        g = _gain(pos, self.E, self.lam, self.model)
        gw = self._P(g / sigma[None, :])
        h, *_ = np.linalg.lstsq(gw.T, Wt, rcond=None)
        out = []
        for k in range(pos.shape[0]):
            amp = float(np.max(np.abs(g[k][:, None] * h[k][None, :]) * 1.0))
            en = float(np.sum((gw[k][:, None] * h[k][None, :]) ** 2))
            out.append(SourceFit(float(pos[k, 0]), float(pos[k, 1]), en, amp))
        return out


class SpaceTimeLocalizer:
    """Single-source engine using the K=1 variable-projection fit (primary source = higher energy)."""

    method = "spacetime_vp"

    def __init__(self, array: ElectrodeArray, delta_threshold: float | None = None, **kw) -> None:
        self.fitter = SpaceTimeFitter(array, **kw)
        self.delta_threshold = delta_threshold

    def estimate(self, snippet: Snippet, use_two: bool = False) -> LocalizationEstimate:
        t0 = time.perf_counter()
        out = self.fitter.fit(snippet)
        if out is None:
            return finish_unavailable(method=self.method, snippet=snippet, t_start=t0, reason="zero signal")
        s = out.sources1[0]
        if use_two and self.delta_threshold is not None and out.delta > self.delta_threshold:
            s = max(out.sources2, key=lambda q: q.energy)
        return finish_estimate(method=self.method, xyz_um=(s.x, s.y, 0.0), snippet=snippet, t_start=t0,
                               confidence=0.5, reason="variable-projection space-time fit")
