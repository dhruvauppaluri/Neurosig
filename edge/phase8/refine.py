"""Refined quasi-static localizer: temporal SVD amplitudes + noise-weighted profile-likelihood fit.

Pipeline per event:
 1. Robust per-channel noise from first differences (signal-insensitive).
 2. Whitened temporal rank-1 projection -> per-channel amplitude a_c with noise sigma_c
    (replaces the single-sample trough; tolerant of +-samples of alignment jitter).
 3. Profile likelihood over (x, y) on a coarse grid with free amplitude A, then bounded
    Levenberg-Marquardt refinement. z is unidentifiable on a collinear probe and fixed at 0.
 4. Gauss-Newton covariance, chi-square residual, and quality flags for later calibration.

Not a beamformer; same quasi-static monopole family as the existing scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import least_squares

from edge.forward import LAMBDA_UM
from edge.localize.common import finish_estimate, finish_unavailable
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet


def channel_noise(voltages: np.ndarray) -> np.ndarray:
    d = np.diff(voltages, axis=0)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0) / 0.6745
    return np.maximum(mad / np.sqrt(2.0), 1e-6)


def _gain(points_xy: np.ndarray, electrodes: np.ndarray, lam: float, model: str) -> np.ndarray:
    """points_xy [P,2] (x,y), z=0 -> [P,C]."""
    dx = points_xy[:, None, 0] - electrodes[None, :, 0]
    dy = points_xy[:, None, 1] - electrodes[None, :, 1]
    dz = electrodes[None, :, 2]
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    if model == "offset_reciprocal":
        return 1.0 / (r / 1000.0 + lam / 1000.0)
    return 1.0 / np.sqrt(r * r + lam * lam)


@dataclass
class FitResult:
    x: float
    y: float
    amp: float
    chi2_red: float
    cov_xy: np.ndarray
    residual_rms: float
    second_peak_ratio: float
    n_used: int
    flags: list = field(default_factory=list)


def temporal_amplitudes(voltages: np.ndarray, sigma: np.ndarray, top_k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel amplitude (positive = trough depth) and its sigma via whitened rank-1 temporal filter."""
    w = voltages / sigma[None, :]
    energy = np.sum(w * w, axis=0)
    order = np.argsort(energy)[::-1][: max(3, min(top_k, w.shape[1]))]
    u, s, vt = np.linalg.svd(w[:, order], full_matrices=False)
    ut = u[:, 0]
    # sign: trough negative -> amplitude positive
    proj_all = ut @ w  # whitened units
    amps = -proj_all * sigma  # back to uV
    if np.sum(amps[order]) < 0:
        amps = -amps
    # noise on amps: unit-norm filter on raw channel = sigma_c
    return amps, sigma


class RefinedScanLocalizer:
    method = "refined_scan"

    def __init__(self, array: ElectrodeArray, lambda_um: float = LAMBDA_UM, model: str = "offset_reciprocal",
                 grid_step_um: tuple[float, float] = (6.0, 10.0), top_k: int = 12, refine: bool = True,
                 use_temporal_svd: bool = True, max_x_um: float = 120.0, clip_negative: bool = False,
                 robust_loss: str = "linear", y_margin_um: float = 40.0, model_error_frac: float = 0.0,
                 spatial_radius_um: float | None = None, prior_um: tuple[float, float] | None = None) -> None:
        self.array = array
        self.lam = float(lambda_um)
        self.model = model
        self.electrodes = array.xyz_um()
        self.top_k = top_k
        self.refine = refine
        self.use_temporal_svd = use_temporal_svd
        self.max_x = max_x_um
        ys = self.electrodes[:, 1]
        xs = self.electrodes[:, 0]
        planar = float(np.ptp(xs)) > 1e-6
        self.x_lo = float(xs.min()) - 0.5 * max_x_um if planar else 1.0
        self.x_hi = float(xs.max()) + 0.5 * max_x_um if planar else max_x_um
        self.clip_negative = clip_negative
        self.prior_um = prior_um   # weak Gaussian prior (x, y) around the strongest contact (data-derived, not sorter)
        self.model_error_frac = model_error_frac   # sigma_eff^2 = sigma^2 + (frac*amp)^2: tolerates spatial-decay mismatch
        self.spatial_radius = spatial_radius_um    # Gaussian window around the strongest channel (real arrays)
        self.robust_loss = robust_loss
        self.y_margin = y_margin_um
        gx = np.arange(self.x_lo, self.x_hi + 0.1, grid_step_um[0])
        gy = np.arange(ys.min() - y_margin_um, ys.max() + y_margin_um + 0.1, grid_step_um[1])
        self.grid = np.array(np.meshgrid(gx, gy, indexing="ij")).reshape(2, -1).T
        self.G = _gain(self.grid, self.electrodes, self.lam, model)  # [P,C]

    # -- amplitudes -------------------------------------------------------------------------
    def observe(self, snippet: Snippet) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(snippet.voltages_uv, float)
        sigma = channel_noise(v)
        if self.use_temporal_svd:
            amps, sig = temporal_amplitudes(v, sigma, self.top_k)
            if self.clip_negative:
                amps = np.maximum(amps, 0.0)
        else:
            idx = min(max(snippet.n_pre, 0), v.shape[0] - 1)
            amps, sig = np.maximum(-v[idx], 0.0), sigma
        sig = np.sqrt(sig ** 2 + (self.model_error_frac * amps) ** 2)
        if self.spatial_radius is not None:
            pk = int(np.argmax(amps))
            d = np.linalg.norm(self.electrodes - self.electrodes[pk], axis=1)
            sig = sig / np.maximum(np.exp(-0.5 * (d / self.spatial_radius) ** 2), 0.05)
        return amps, sig

    def _profile(self, a: np.ndarray, w: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best amplitude and weighted SSE for each grid row; w = 1/sigma^2 (zeroed for excluded channels)."""
        num = (G * (a * w)[None, :]).sum(axis=1)
        den = (G * G * w[None, :]).sum(axis=1)
        amp = np.maximum(num / np.maximum(den, 1e-18), 0.0)
        resid = a[None, :] - amp[:, None] * G
        sse = (resid * resid * w[None, :]).sum(axis=1)
        return amp, sse

    def fit(self, snippet: Snippet) -> FitResult | None:
        a, sigma = self.observe(snippet)
        w = 1.0 / np.square(sigma)
        if float(np.max(a)) <= 0:
            return None
        amp0, sse = self._profile(a, w, self.G)
        pk = int(np.argmax(a))
        if self.prior_um is not None:
            c = self.electrodes[pk]
            sse = sse + ((self.grid[:, 0] - c[0]) / self.prior_um[0]) ** 2 + ((self.grid[:, 1] - c[1]) / self.prior_um[1]) ** 2
        order = np.argsort(sse)
        best = int(order[0])
        # second peak: best cell at least 60 um away in y or 25 in x
        far = np.abs(self.grid[:, 1] - self.grid[best, 1]) > 60.0
        second = float(sse[far].min()) if np.any(far) else float(sse[best]) * 4
        ratio = float(sse[best] / max(second, 1e-12))
        x0 = np.array([self.grid[best, 0], self.grid[best, 1]])
        A0 = float(amp0[best])
        n = a.size
        x = x0
        A = A0
        cov = np.full((2, 2), np.inf)
        if self.refine:
            def res(p):
                g = _gain(p[None, :2], self.electrodes, self.lam, self.model)[0]
                r = (a - p[2] * g) / sigma
                if self.prior_um is not None:
                    c = self.electrodes[pk]
                    r = np.concatenate([r, [(p[0] - c[0]) / self.prior_um[0], (p[1] - c[1]) / self.prior_um[1]]])
                return r
            lo = np.array([self.x_lo - 0.5, self.electrodes[:, 1].min() - self.y_margin, 0.0])
            hi = np.array([self.x_hi, self.electrodes[:, 1].max() + self.y_margin, 10 * max(A0, 1.0)])
            p0 = np.clip(np.array([x0[0], x0[1], A0]), lo + 1e-6, hi - 1e-6)
            sol = least_squares(res, p0, bounds=(lo, hi), method="trf", x_scale=[10.0, 10.0, max(A0, 1.0)],
                                loss=self.robust_loss, f_scale=2.0)
            x, A = sol.x[:2], float(sol.x[2])
            J = sol.jac
            dof = max(n - 3, 1)
            chi2 = float(np.sum(sol.fun ** 2))
            try:
                full = np.linalg.pinv(J.T @ J)
                cov = full[:2, :2] * max(chi2 / dof, 1.0)
            except np.linalg.LinAlgError:
                cov = np.full((2, 2), np.inf)
        g = _gain(x[None, :], self.electrodes, self.lam, self.model)[0]
        r = (a - A * g) / sigma
        chi2_red = float(np.sum(r * r) / max(n - 3, 1))
        return FitResult(float(x[0]), float(x[1]), float(A), chi2_red, cov, float(np.sqrt(np.mean(r * r))),
                         ratio, n)

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        t0 = time.perf_counter()
        fit = self.fit(snippet)
        if fit is None:
            return finish_unavailable(method=self.method, snippet=snippet, t_start=t0, reason="zero amplitude")
        conf = float(np.clip(1.0 / (1.0 + max(fit.chi2_red - 1.0, 0.0)), 0.0, 1.0))
        return finish_estimate(method=self.method, xyz_um=(fit.x, fit.y, 0.0), snippet=snippet, t_start=t0,
                               confidence=conf, reason="temporal-SVD amplitudes + weighted profile likelihood + LM")
