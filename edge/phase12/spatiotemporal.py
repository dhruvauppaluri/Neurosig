"""Phase 12.3: spatiotemporal amplitude extraction and fitting for single-event localization.

Phase 11 showed that a 2-sample timing shift moves a clean event's estimate by ~12 um and that noise explains about
half of the within-unit deviation: the single-sample trough amplitude vector is the weak link, not the spatial model
(11B). The variants below change only how the per-contact amplitude vector is formed from a short multichannel
window, or how the fit is weighted, and keep the line20 forward model fixed:

  trough     E3 trough row (Phase 11B input; raw - median CAR, pre-event baseline removed)
  mean3      mean of the three samples around the trough
  rank1      rank-1 SVD denoising of the noise-whitened window [-0.3, +0.6] ms; amplitude = the denoised trough row
  mf         matched filter: every contact projected on the peak contact's own waveform over the same window
  rank1_gls  rank1 amplitudes, generalized least squares with a shrunk spatial noise covariance (event-free samples)
  rank1_sel  rank1 amplitudes on adaptively selected contacts (|a|/sigma >= 1.5 or within 60 um of the peak contact)
  st_vp      Phase 9/10 K = 1 full-waveform space-time VP fit (free temporal waveform, rank 4, common-mode nuisance)
  linefree_map  rank1 amplitudes, line source with a free distance h and a Gaussian prior h ~ N(20, 10^2) um
                (position and footprint-width parameters regularized separately)

Identifiability: from the line20 grid, the area of the profile-likelihood region delta-SSE <= 2.3 max(chi2_red, 1)
(an approximate 68% joint region for two position parameters). It is an output, not a filter.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from edge.phase11 import forward_models as F

WIN = (-9, 18)          # samples around the aligned trough (-0.3, +0.6 ms at 30 kHz)
SHRINK = 0.3
H_PRIOR = (20.0, 10.0)
VARIANTS = ("trough", "mean3", "rank1", "mf", "rank1_gls", "rank1_sel", "linefree_map")
HP_VARIANTS = ("trough_hp", "mean3_hp", "rank1_hp", "mf_hp", "rank1_gls_hp", "rank1_sel_hp")
HP_BOX = 31             # samples (~1 ms): centred moving-average removal = crude high-pass (~0.5-1 kHz)


def window(rc: np.ndarray, k: int, hp: bool = False) -> np.ndarray:
    """Baseline-subtracted raw-CAR window [WIN] around sample k of a [T, C] segment. ``hp`` first removes a centred
    31-sample moving average per contact (slow LFP / drift components that a rank-1 fit would otherwise capture)."""
    if hp:
        from scipy.ndimage import uniform_filter1d
        rc = rc - uniform_filter1d(rc, HP_BOX, axis=0, mode="nearest")
    base = np.median(rc[k - 40:k - 12], axis=0)
    return rc[k + WIN[0]:k + WIN[1]] - base[None]


def base_name(v: str) -> str:
    return v[:-3] if v.endswith("_hp") else v


def amplitudes(W: np.ndarray, sigma: np.ndarray, method: str) -> np.ndarray:
    method = base_name(method)
    k0 = -WIN[0]
    if method == "trough":
        return -W[k0]
    if method == "mean3":
        return -W[k0 - 1:k0 + 2].mean(axis=0)
    if method in ("rank1", "rank1_gls", "rank1_sel", "linefree_map"):
        Wn = W / sigma[None]
        U, S, Vt = np.linalg.svd(Wn, full_matrices=False)
        r1 = S[0] * np.outer(U[:, 0], Vt[0])
        return -(r1[k0] * sigma)
    if method == "mf":
        pk = int(np.argmax(-W[k0] / sigma))
        h = W[:, pk]
        return -(W.T @ h) / max(h @ h, 1e-12) * h[k0]
    raise ValueError(method)


def noise_cov(rc_list: list[np.ndarray], ks: list[int]) -> np.ndarray:
    """Shrunk spatial covariance of event-free pre-event samples (s-40 .. s-12) pooled over events of one window."""
    X = np.concatenate([rc[k - 40:k - 12] - np.median(rc[k - 40:k - 12], axis=0) for rc, k in zip(rc_list, ks)])
    S = np.cov(X.T)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))


class GLSLine20:
    """line20 VP fit with a full noise covariance (whitening by the inverse Cholesky factor)."""

    def __init__(self, E: np.ndarray, cov: np.ndarray, y_margin=40.0, x_range=(-20.0, 80.0)):
        self.E = E
        self.L = np.linalg.inv(np.linalg.cholesky(cov))       # whitening: L @ a
        self.P = F._grid(E, y_margin, x_range)
        self.G = F._grid_gains(F.Model("line20"), self.P, E) @ self.L.T     # [P, C] whitened gains
        self.lo = np.array([x_range[0], E[:, 1].min() - y_margin]); self.hi = np.array([x_range[1], E[:, 1].max() + y_margin])

    def fit(self, a: np.ndarray) -> np.ndarray:
        aw = self.L @ a
        num = self.G @ aw; den = np.sum(self.G ** 2, axis=1)
        sc = np.where(num > 0, num ** 2 / np.maximum(den, 1e-18), 0.0)
        th0 = self.P[int(np.argmax(sc))]

        def res(th):
            g = self.L @ F.line_gain(th[0], th[1], self.E, 20.0)
            amp = max(float(g @ aw) / max(float(g @ g), 1e-18), 0.0)
            return aw - amp * g
        return least_squares(res, np.clip(th0, self.lo + 1e-6, self.hi - 1e-6), bounds=(self.lo, self.hi), method="trf",
                             x_scale=10.0, max_nfev=40).x


def linefree_map(a: np.ndarray, sigma: np.ndarray, E: np.ndarray, y_margin=40.0, x_range=(-20.0, 80.0)) -> np.ndarray:
    w = 1.0 / sigma
    f0 = F.fit(F.Model("line20"), a, sigma, E, y_margin=y_margin, x_range=x_range)
    mu, sd = H_PRIOR

    def res(th):
        g = F.line_gain(th[0], th[1], E, th[2]) * w
        aw = a * w
        amp = max(float(g @ aw) / max(float(g @ g), 1e-18), 0.0)
        return np.concatenate([aw - amp * g, [(th[2] - mu) / sd]])
    lo = np.array([x_range[0], E[:, 1].min() - y_margin, 5.0]); hi = np.array([x_range[1], E[:, 1].max() + y_margin, 80.0])
    x0 = np.clip(np.array([f0.theta[0], f0.theta[1], 20.0]), lo + 1e-6, hi - 1e-6)
    return least_squares(res, x0, bounds=(lo, hi), method="trf", x_scale=10.0, max_nfev=60).x[:2]


def identifiability_area(a: np.ndarray, sigma: np.ndarray, E: np.ndarray, cache: dict, step=8.0) -> float:
    """Area (um^2) of the grid region with delta-SSE <= 2.3 * max(chi2_red, 1) under line20 (approximate 68% region)."""
    key = ("ident", E.tobytes())
    if key not in cache:
        P = F._grid(E, 40.0, (-20.0, 80.0), step); cache[key] = (P, F._grid_gains(F.Model("line20"), P, E))
    P, G = cache[key]
    w = 1.0 / np.square(sigma)
    num = G @ (a * w); den = (G * G) @ w; tot = float(np.sum(a * a * w))
    amp = np.maximum(num / np.maximum(den, 1e-18), 0.0)
    sse = tot - 2 * amp * num + amp * amp * den
    m = float(sse.min()); chi = max(m / max(len(a) - 3, 1), 1.0)
    return float(np.sum(sse - m <= 2.3 * chi) * step * step)


def localize_variant(method: str, a: np.ndarray, sigma: np.ndarray, E: np.ndarray, cache: dict, gls: GLSLine20 | None = None):
    method = base_name(method)
    if method == "rank1_gls":
        return gls.fit(a)
    if method == "linefree_map":
        return linefree_map(a, sigma, E)
    mask = None
    if method == "rank1_sel":
        pk = int(np.argmax(a / sigma))
        mask = (a / sigma >= 1.5) | (np.linalg.norm(E[:, :2] - E[pk, :2], axis=1) <= 60.0)
        if mask.sum() < 6:
            mask = None
    f = F.fit(F.Model("line20"), a, sigma, E, mask=mask, cache=cache)
    return f.theta[:2].copy() if f.amps[0] > 0 else None
