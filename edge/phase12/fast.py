"""Phase 12.6: fast single-source path and the frozen P12 event localizer.

FastLine20 reproduces the line20 variable-projection fit (edge/phase11/forward_models.fit with Model('line20')) without
scipy's least_squares: a cached 4 um grid (gains and per-sigma denominators precomputed per contact set), then three
Gauss-Newton steps with the analytic Jacobian of the projected residual, bounded to the search box. The same grid
gives the identifiability area for free. Equivalence with the reference fit is a regression test.

P12Localizer (frozen configuration P12):
  single source   E3 alignment (raw - median CAR, +-3 samples over contacts within 60 um), mean of 3 samples around
                  the trough, FastLine20; identifiability area as the uncertainty input (12.3/12.5)
  count           cheap gate (rank2, n_trough, t_spread, line20 chi2_red, SNR: logistic, p12dev) decides whether the
                  Phase 9/10 K<=2 analysis runs; if it runs, the P12-LR rule (count_model_p12.json) decides 1 vs 2
  unit level      pooling (edge/phase12/pooling.py, l20_robust) is a separate, offline / sorter-assisted path
"""

from __future__ import annotations

import numpy as np

from edge.phase11 import forward_models as F

LAM = 20.0


class FastLine20:
    def __init__(self, E: np.ndarray, y_margin=40.0, x_range=(-20.0, 80.0), step=4.0):
        self.E = E[:, :2].astype(float)
        self.P = F._grid(E, y_margin, x_range, step)
        self.G = F._grid_gains(F.Model("line20"), self.P, E)            # [P, C]
        self.G2 = self.G * self.G
        self.lo = np.array([x_range[0], E[:, 1].min() - y_margin]); self.hi = np.array([x_range[1], E[:, 1].max() + y_margin])
        self.step = step
        self._den = {}

    def _gain_jac(self, th):
        dx = th[0] - self.E[:, 0]; dy = th[1] - self.E[:, 1]
        q = dx * dx + dy * dy + LAM * LAM
        g = LAM / np.sqrt(q)
        dg = -LAM * q[:, None] ** -1.5 * np.stack([dx, dy], axis=1)       # [C, 2]
        return g, dg

    def fit(self, a: np.ndarray, sigma: np.ndarray, iters: int = 3, area: bool = False):
        w = 1.0 / np.square(sigma)
        key = sigma.tobytes()
        den = self._den.get(key)
        if den is None:
            den = self.G2 @ w
            if len(self._den) > 8:
                self._den.clear()
            self._den[key] = den
        aw = a * w
        num = self.G @ aw
        amp = np.maximum(num / np.maximum(den, 1e-18), 0.0)
        sc = amp * num                                      # = tot - sse
        i = int(np.argmax(sc))
        if sc[i] <= 0:
            return None, None, None
        th = self.P[i].copy()
        sw = np.sqrt(w); asw = a * sw

        def sse(t):
            g = LAM / np.sqrt((t[0] - self.E[:, 0]) ** 2 + (t[1] - self.E[:, 1]) ** 2 + LAM * LAM) * sw
            A = max(float(g @ asw) / float(g @ g), 0.0)
            r = asw - A * g
            return float(r @ r)
        cur = sse(th)
        radius = self.step                  # trust radius: never leave the neighbourhood of the grid optimum blindly
        for _ in range(iters):
            g, dg = self._gain_jac(th)
            gw = g * sw; dgw = dg * sw[:, None]
            gg = gw @ gw; A = max(float(gw @ asw) / gg, 0.0)
            r = asw - A * gw
            # Jacobian of the VP residual (Kaufman approximation): -A * (I - u u^T) dg
            u = gw / np.sqrt(gg)
            J = -A * (dgw - np.outer(u, u @ dgw))
            JTJ = J.T @ J
            d = -np.linalg.solve(JTJ + 1e-6 * np.trace(JTJ) * np.eye(2) + 1e-12 * np.eye(2), J.T @ r)
            nd = float(np.linalg.norm(d))
            if nd > radius:
                d *= radius / nd
            cand = np.clip(th + d, self.lo, self.hi)
            new = sse(cand)
            if new < cur:
                th, cur = cand, new
            else:
                radius *= 0.5
        tot = float(asw @ asw)
        sse_min = tot - float(sc[i])
        chi = max(sse_min / max(len(a) - 3, 1), 1.0)
        ar = float(np.sum((tot - sc) - sse_min <= 2.3 * chi) * self.step * self.step) if area else None
        return th, ar, sse_min / max(len(a) - 3, 1)


def e3_align(rc: np.ndarray, s: int, ch: int, E: np.ndarray, sig: np.ndarray, half: int = 3, radius: float = 60.0) -> int:
    S = np.flatnonzero(np.linalg.norm(E[:, :2] - E[ch, :2], axis=1) <= radius)
    lo, hi = max(s - half, 0), min(s + half + 1, rc.shape[0])
    base = np.median(rc[max(s - 40, 0):max(s - 12, 1)][:, S], axis=0)
    return lo + int(np.argmin(((rc[lo:hi][:, S] - base) / sig[S]).sum(axis=1)))


def mean3_amplitudes(rc: np.ndarray, k: int) -> np.ndarray:
    base = np.median(rc[k - 40:k - 12], axis=0)
    return -(rc[k - 1:k + 2].mean(axis=0) - base)


GATE_FEATURES = ("rank2", "n_trough", "t_spread", "log_chi_l20", "log_snr")


def cheap_features(v: np.ndarray, k: int, fl: FastLine20, sigma_fit: np.ndarray):
    """Single-source estimate + gate features from one raw snippet (no K=2 fitting). Returns (theta, area, feats)."""
    from edge.phase8.refine import channel_noise
    from edge.phase12.overlap_bench import temporal_features
    from edge.types import Snippet
    rc = v - np.median(v, axis=1, keepdims=True)
    b0 = max(k - 40, 0); b1 = max(k - 12, b0 + 4)
    base = np.median(rc[b0:b1], axis=0)
    a = -(rc[k - 1:k + 2].mean(axis=0) - base)
    th, ar, chi = fl.fit(a, sigma_fit, iters=5, area=True)
    sig = channel_noise(v)
    sn = Snippet(0, k, 0, 30000.0, tuple(range(v.shape[1])), v, np.c_[fl.E, np.zeros(len(fl.E))], None, None)
    tf = temporal_features(sn, sig)
    snr = float(np.max(a / np.maximum(sig, 1e-6)))
    feats = np.array([tf["rank2"], min(tf["n_trough"], 4), min(tf["t_spread"], 10.0), np.log(max(chi or 1e-6, 1e-6)), np.log(max(snr, 1e-3))])
    return th, ar, feats


class CheapGate:
    """Logistic pre-screen on GATE_FEATURES predicting 'the P12-LR rule would declare two sources'; threshold keeps
    >= 99% of those events on p12dev. Fitted by scripts/phase12/run_12_6.py; stored in reports/phase12/gate_p12.json."""

    def __init__(self, d: dict):
        self.mu, self.sd, self.w, self.thr = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"]), float(d["thr"])

    def prob(self, f):
        z = (np.atleast_2d(f) - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(self.w[0] + z @ self.w[1:])))

    def invoke(self, f) -> bool:
        return bool(self.prob(f)[0] >= self.thr)


# ------------------------------------------------------------------------------------------------ fast K<=2 refine
from edge.phase8.multisource import SpaceTimeFitter  # noqa: E402
from edge.phase8.refine import _gain  # noqa: E402


class FastSpaceTimeFitter(SpaceTimeFitter):
    """Same model, grid, bounds and statistics as SpaceTimeFitter; only ``_refine`` changes: scipy least_squares
    (trf, numerical Jacobian) is replaced by a trust-region Gauss-Newton with the analytic Kaufman variable-projection
    Jacobian of the line-source gains. Agreement with the reference fitter is measured in run_12_6.py (equivalence)."""

    ITERS = 10

    def _shared_lambda(self, g, R):
        """Closed-form 2x2 version of SpaceTimeFitter._shared_lambda (same eigenpair selection: the largest
        generalized eigenvalue whose eigenvector has same-sign components)."""
        a11 = float(g[0] @ g[0]) + 1e-12; a22 = float(g[1] @ g[1]) + 1e-12; a12 = float(g[0] @ g[1])
        Rg0 = R @ g[0]; Rg1 = R @ g[1]
        b11 = float(g[0] @ Rg0); b22 = float(g[1] @ Rg1); b12 = float(g[0] @ Rg1)
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-300:
            return -1.0, np.ones(2)
        n11 = (a22 * b11 - a12 * b12) / det; n12 = (a22 * b12 - a12 * b22) / det
        n21 = (-a12 * b11 + a11 * b12) / det; n22 = (-a12 * b12 + a11 * b22) / det
        tr = n11 + n22; dsc = np.sqrt(max((n11 - n22) ** 2 / 4 + n12 * n21, 0.0))
        best, coef = -1.0, np.ones(2)
        for lam in (tr / 2 + dsc, tr / 2 - dsc):
            if n12 * (lam - n11) > 0 and lam > best:
                best, coef = float(lam), np.array([n12, lam - n11])
        return best, coef

    def _gains_jac(self, pos):
        dx = pos[:, None, 0] - self.E[None, :, 0]; dy = pos[:, None, 1] - self.E[None, :, 1]; dz = self.E[None, :, 2]
        q = dx * dx + dy * dy + dz * dz + self.lam * self.lam
        g = q ** -0.5
        return g, -dx * q ** -1.5, -dy * q ** -1.5                    # [K, C] each

    def _refine(self, pos, Wt, sigma):
        k = pos.shape[0]
        lo, hi = np.tile(self.lo, k).reshape(k, 2), np.tile(self.hi, k).reshape(k, 2)
        p = np.clip(pos.copy(), lo + 1e-6, hi - 1e-6)
        cur = self._vp_sse(p, Wt, sigma)
        radius = 10.0
        for _ in range(self.ITERS):
            g, gx, gy = self._gains_jac(p)
            Gw = self._P(g / sigma[None]); Dx = self._P(gx / sigma[None]); Dy = self._P(gy / sigma[None])
            h, *_ = np.linalg.lstsq(Gw.T, Wt, rcond=None)                  # [K, r]
            R = Wt - Gw.T @ h                                              # [C, r]
            Q, _ = np.linalg.qr(Gw.T)                                      # projector onto span of the gains
            cols = []
            for j in range(k):
                for D in (Dx, Dy):
                    M = np.outer(D[j], h[j])                               # d(G^T h)/d p_jd  [C, r]
                    cols.append(-(M - Q @ (Q.T @ M)).ravel())
            J = np.stack(cols, axis=1)
            JTJ = J.T @ J
            d = -np.linalg.solve(JTJ + 1e-6 * np.trace(JTJ) * np.eye(2 * k) + 1e-12 * np.eye(2 * k), J.T @ R.ravel())
            nd = float(np.linalg.norm(d))
            if nd > radius:
                d *= radius / nd
            cand = np.clip(p + d.reshape(k, 2), lo, hi)
            new = self._vp_sse(cand, Wt, sigma)
            if new < cur:
                p, cur = cand, new
                if nd < 0.05:
                    break
            else:
                radius *= 0.5
                if radius < 0.05:
                    break
        return p


def fast_localizer(array, cal=None):
    """The frozen Phase 9/10 localizer configuration with the fast refine (P12 runtime)."""
    from edge.phase10.overlap_stress import frozen_calibration
    from edge.phase8.pipeline import Phase8Localizer
    loc = Phase8Localizer(array, cal or frozen_calibration(), model="line_source", lambda_um=20.0, rank=4,
                          max_x_um=90.0, y_margin_um=40.0, nuisance_cm=True)
    f = loc.fitter
    ff = FastSpaceTimeFitter.__new__(FastSpaceTimeFitter)
    ff.__dict__.update(f.__dict__)
    loc.fitter = ff
    return loc
