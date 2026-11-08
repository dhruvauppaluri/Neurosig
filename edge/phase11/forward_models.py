"""Phase 11B spatial forward models for the per-contact trough amplitude vector, and a generic variable-projection
fitter (amplitudes linear, positions nonlinear).

Models (gain g_c(theta) per contact; amplitude(s) solved by weighted least squares):
  line20     A: P9 baseline, line source 1/sqrt(r^2 + 20^2) (equivalently a point source 20 um off the plane)
  linefree   A2: line source with a per-event free distance h (theta = x, y, h), h in [5, 80] um
  radial     B: empirical radial footprint f(r_xy), monotone, learned from training units (piecewise-linear table)
  aniso      D: empirical anisotropic footprint f(|dx|, dy) on a grid, learned from training units (bilinear)
  twoline    C: two line sources (lambda 20) with independent signed amplitudes (theta = x1, y1, x2, y2)
Positions are in the probe plane (x lateral, y axial). z / distance is never claimed as a coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

R_KNOTS = np.array([0, 8, 16, 24, 32, 45, 60, 80, 110, 160, 250, 400], float)
DX_KNOTS = np.array([0, 8, 16, 24, 32, 48, 64, 96, 400], float)
DY_KNOTS = np.array([-400, -160, -110, -80, -60, -45, -32, -24, -16, -8, 0, 8, 16, 24, 32, 45, 60, 80, 110, 160, 400], float)


def line_gain(px, py, E, lam):
    r2 = (E[:, 0] - px) ** 2 + (E[:, 1] - py) ** 2
    return lam / np.sqrt(r2 + lam * lam)


@dataclass
class Model:
    name: str
    n_src: int = 1
    radial: np.ndarray | None = None         # values at R_KNOTS (B)
    aniso: np.ndarray | None = None          # [len(DX), len(DY)] (D)
    _interp: object = field(default=None, repr=False)

    def n_theta(self):
        return {"line20": 2, "line20c": 2, "linefree": 3, "radial": 2, "aniso": 2, "twoline": 4}[self.name]

    def gains(self, theta, E) -> np.ndarray:
        """[n_src, C]"""
        if self.name == "line20":
            return line_gain(theta[0], theta[1], E, 20.0)[None]
        if self.name == "line20c":           # E: line source + signed constant (common-mode / far-field offset)
            return np.stack([line_gain(theta[0], theta[1], E, 20.0), np.ones(len(E))])
        if self.name == "linefree":
            return line_gain(theta[0], theta[1], E, theta[2])[None]
        if self.name == "twoline":
            return np.stack([line_gain(theta[0], theta[1], E, 20.0), line_gain(theta[2], theta[3], E, 20.0)])
        if self.name == "radial":
            r = np.hypot(E[:, 0] - theta[0], E[:, 1] - theta[1])
            return np.interp(r, R_KNOTS, self.radial)[None]
        if self.name == "aniso":
            if self._interp is None:
                self._interp = RegularGridInterpolator((DX_KNOTS, DY_KNOTS), self.aniso, bounds_error=False, fill_value=0.0)
            dx = np.abs(E[:, 0] - theta[0]); dy = E[:, 1] - theta[1]
            return self._interp(np.stack([np.clip(dx, 0, 400), np.clip(dy, -400, 400)], axis=1))[None]
        raise ValueError(self.name)


def _vp(theta, model, a, w, E):
    G = model.gains(theta, E)                    # [K, C]
    Gw = G * np.sqrt(w)[None]
    aw = a * np.sqrt(w)
    h, *_ = np.linalg.lstsq(Gw.T, aw, rcond=None)
    if model.n_src == 1:
        h = np.maximum(h, 0.0)
    elif model.name == "line20c" and h[0] < 0:
        h = np.array([0.0, float(np.sum(aw * Gw[1]) / max(np.sum(Gw[1] ** 2), 1e-18))])
    return aw - Gw.T @ h, h


@dataclass
class Fit:
    theta: np.ndarray
    amps: np.ndarray
    sse: float
    sse_rel: float


def _grid(Em, y_margin, x_range, step=8.0):
    ys = Em[:, 1]
    gx = np.arange(x_range[0], x_range[1] + 0.1, step)
    gy = np.arange(ys.min() - y_margin, ys.max() + y_margin + 0.1, step)
    return np.array(np.meshgrid(gx, gy, indexing="ij")).reshape(2, -1).T


def _grid_gains(model, P, Em, h=None):
    if model.name in ("line20", "twoline", "line20c"):
        lam = 20.0
    elif model.name == "linefree":
        lam = h
    else:
        lam = None
    if lam is not None:
        r2 = (P[:, None, 0] - Em[None, :, 0]) ** 2 + (P[:, None, 1] - Em[None, :, 1]) ** 2
        return lam / np.sqrt(r2 + lam * lam)
    return np.concatenate([model.gains(p, Em) for p in P], axis=0)


def fit(model: Model, a: np.ndarray, sigma: np.ndarray, E: np.ndarray, mask: np.ndarray | None = None,
        y_margin=40.0, x_range=(-20.0, 80.0), cache: dict | None = None) -> Fit:
    """Weighted VP fit of the trough amplitude vector ``a`` (positive = trough depth) on contacts ``mask``.
    Vectorized grid initialisation (cached per contact set via ``cache``), then bounded LM refinement."""
    m = np.ones(len(a), bool) if mask is None else np.asarray(mask, bool)
    Em, am, wm = E[m], a[m], 1.0 / np.square(sigma[m])
    key = (model.name, m.tobytes(), E.tobytes())
    if cache is not None and key in cache:
        P, Gs = cache[key]
    else:
        P = _grid(Em, y_margin, x_range)
        hs = (10.0, 20.0, 40.0) if model.name == "linefree" else (None,)
        Gs = [(h, _grid_gains(model if model.name not in ("twoline", "line20c") else Model("line20"), P, Em, h)) for h in hs]
        if cache is not None:
            cache[key] = (P, Gs)
    tot = float(np.sum(am * am * wm))
    best, bt = np.inf, None
    for h, G in Gs:
        num = G @ (am * wm); den = (G * G) @ wm
        amp = np.maximum(num / np.maximum(den, 1e-18), 0.0)
        sse = tot - 2 * amp * num + amp * amp * den
        i = int(np.argmin(sse))
        if sse[i] < best:
            best, bt = float(sse[i]), np.array(list(P[i]) + ([h] if h is not None else []))
    if model.name == "twoline":
        G = Gs[0][1]; g1 = _grid_gains(Model("line20"), bt[None, :2], Em)[0]
        a11 = (g1 * g1) @ wm; a12 = G @ (g1 * wm); a22 = (G * G) @ wm
        b1 = (g1 * wm) @ am; b2 = G @ (am * wm)
        det = a11 * a22 - a12 ** 2 + 1e-18
        h1 = (a22 * b1 - a12 * b2) / det; h2 = (a11 * b2 - a12 * b1) / det
        sse = tot - (h1 * b1 + h2 * b2)
        sse[det < 1e-6 * a11 * a22] = np.inf
        j = int(np.argmin(sse)); bt = np.array([bt[0], bt[1], P[j, 0], P[j, 1]])
    ys = Em[:, 1]
    lo = [x_range[0], ys.min() - y_margin] * (2 if model.name == "twoline" else 1)
    hi = [x_range[1], ys.max() + y_margin] * (2 if model.name == "twoline" else 1)
    if model.name == "linefree":
        lo, hi = lo + [5.0], hi + [80.0]
    lo, hi = np.array(lo), np.array(hi)
    sol = least_squares(lambda th: _vp(th, model, am, wm, Em)[0], np.clip(bt, lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                        method="trf", x_scale=10.0, max_nfev=40)
    r, h = _vp(sol.x, model, am, wm, Em)
    return Fit(sol.x, h, float(r @ r), float(r @ r) / max(tot, 1e-12))


def predict(model: Model, f: Fit, E: np.ndarray) -> np.ndarray:
    return f.amps @ model.gains(f.theta, E)


# --------------------------------------------------------------------------------------------- learning tables
def learn_radial(samples) -> np.ndarray:
    """samples: list of (r_xy array, normalized amplitude array). Monotone non-increasing table with f(0)=1."""
    r = np.concatenate([s[0] for s in samples]); v = np.concatenate([s[1] for s in samples])
    vals = []
    for k in range(len(R_KNOTS)):
        lo = R_KNOTS[k - 1] if k else -1; hi = R_KNOTS[k + 1] if k + 1 < len(R_KNOTS) else 1e9
        sel = (r > (lo + R_KNOTS[k]) / 2) & (r <= (R_KNOTS[k] + hi) / 2) if k else (r <= R_KNOTS[1] / 2)
        vals.append(np.median(v[sel]) if sel.sum() >= 10 else np.nan)
    vals = np.array(vals, float)
    for k in range(len(vals)):                       # fill gaps by line-source shape
        if not np.isfinite(vals[k]):
            vals[k] = 20.0 / np.sqrt(R_KNOTS[k] ** 2 + 400.0)
    vals = np.minimum.accumulate(np.clip(vals, 0.0, None)); vals = vals / max(vals[0], 1e-9)
    return vals


def learn_aniso(samples, prior: np.ndarray) -> np.ndarray:
    """samples: list of (dx, dy, normalized amplitude). Cell medians, shrunk toward the radial prior when sparse."""
    dx = np.concatenate([s[0] for s in samples]); dy = np.concatenate([s[1] for s in samples]); v = np.concatenate([s[2] for s in samples])
    T = np.zeros((len(DX_KNOTS), len(DY_KNOTS)))
    ix = np.clip(np.searchsorted(DX_KNOTS, np.abs(dx)), 0, len(DX_KNOTS) - 1)
    iy = np.clip(np.searchsorted(DY_KNOTS, dy), 0, len(DY_KNOTS) - 1)
    for i in range(len(DX_KNOTS)):
        for j in range(len(DY_KNOTS)):
            base = float(np.interp(np.hypot(DX_KNOTS[i], DY_KNOTS[j]), R_KNOTS, prior))
            sel = (ix == i) & (iy == j); n = int(sel.sum())
            T[i, j] = (n * float(np.median(v[sel])) + 20 * base) / (n + 20) if n else base
    return T / max(T[0, np.searchsorted(DY_KNOTS, 0)], 1e-9)
