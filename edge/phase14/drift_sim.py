"""Phase 14.5: ground-truth event streams with probe drift (MEArec/NEURON templates + real IBL background noise).

A stream holds n_units neurons on one fixed 32-contact window. Each neuron is a biophysical template placed at a
lattice offset; probe drift d(t) moves every neuron axially by the same amount (probe displacement), so a neuron's
waveform on the fixed contacts changes over time. Continuous offsets are rendered by linear interpolation between
the two neighbouring exact lattice placements (40 um on NP1), an approximation stated as a limitation. Spikes are
Poisson (rate_hz) and isolated (no overlaps); amplitudes jitter +-10%. Drift profiles:
  none | linear (0 -> amp over the stream) | step (amp at the midpoint) | sine (amp * sin, one period)
Features match the real-data cache (edge/phase14/detcache.py): ``wave`` = median-CAR segment band-passed
300-6000 Hz (zero-phase on the 120-sample segment; the real cache uses the causal frozen filter on the full window),
divided by the per-contact MAD of identically filtered background (samples -15..+29), ``amp3`` E3-style mean-3 trough amplitudes, detector channel =
most negative contact, SNR = trough depth / MAD. Ground truth: unit id and true soma position at each spike.
"""

from __future__ import annotations

import numpy as np

from edge.phase14 import bench14 as B


def _bp(x):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, [300, 6000], btype="bandpass", fs=30000.0, output="sos")
    return sosfiltfilt(sos, x, axis=-2, padlen=40)


def filtered_mad(nb: np.ndarray, n: int = 600) -> np.ndarray:
    x = nb[:n].astype(float); x = x - np.median(x, axis=2, keepdims=True)
    f = _bp(x)[:, 20:-20].reshape(-1, x.shape[2])
    return np.median(np.abs(f - np.median(f, axis=0)), axis=0) / 0.6745


def _profile(kind: str, amp: float, t: np.ndarray, dur: float) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(t)
    if kind == "linear":
        return amp * t / dur
    if kind == "step":
        return np.where(t >= dur / 2, amp, 0.0)
    if kind == "sine":
        return amp * np.sin(2 * np.pi * t / dur)
    raise ValueError(kind)


def make_stream(pool: str, n_units: int = 5, duration_s: float = 120.0, rate_hz: float = 4.0, drift: str = "linear",
                amp_um: float = 30.0, seed: int = 0, snr_range=(6.0, 16.0)) -> dict:
    ts = B.template_set(pool); cfg = B.POOLS[pool]
    nb = B.noise_bank(cfg["noise"]); sig = B.noise_sigma(cfg["noise"]); fmad = filtered_mad(nb)
    rng_mad = None
    rng = np.random.default_rng(seed)
    row = B.ROW[ts.probe]
    k0 = int(rng.integers(len(ts.W)))
    _, E, _, _, _ = B.view(ts, k0)
    ylo, yhi = E[:, 1].min() + 40, E[:, 1].max() - 40
    units = []
    ks = rng.choice(len(ts.W), n_units, replace=False)
    lo_d, hi_d = min(0.0, -abs(amp_um)), max(0.0, abs(amp_um))
    for j, k in enumerate(ks):
        y = ylo + (yhi - ylo) * (j + rng.uniform(0.2, 0.8)) / n_units
        dz0 = round((y - ts.loc[k, 2]) / row) * row
        offs = np.arange(np.floor(lo_d / row) - 1, np.ceil(hi_d / row) + 2) * row
        Ws = {}
        for o in offs:
            W, tr, t0 = B.shifted(ts, k, E, dz0 + o)
            Ws[float(o)] = W
        W0, tr0, t00 = B.shifted(ts, k, E, dz0)
        pk = int(np.argmin(W0[t00]))
        if -W0[t00, pk] <= 1e-6:
            continue
        snr = rng.uniform(*snr_range)
        units.append(dict(k=int(k), Ws=Ws, t0=t00, truth0=tr0, scale=snr * sig[pk] / -W0[t00, pk], cell=str(ts.cell[k])))
    times, who = [], []
    for j in range(len(units)):
        n = rng.poisson(rate_hz * duration_s)
        times += list(np.sort(rng.uniform(0, duration_s, n))); who += [j] * n
    o = np.argsort(times); times = np.array(times)[o]; who = np.array(who)[o]
    d = _profile(drift, amp_um, times, duration_s)
    cols = {k: [] for k in ("wave", "amp3", "ch", "snr", "t", "unit", "true_pos", "drift")}
    for t, j, dd in zip(times, who, d):
        u = units[j]
        lo = np.floor(dd / row) * row; f = (dd - lo) / row
        W = (1 - f) * u["Ws"][float(lo)] + f * u["Ws"][float(lo + row)] if f > 1e-9 else u["Ws"][float(lo)]
        seg = B.place(W * u["scale"] * (1 + 0.1 * rng.uniform(-1, 1)), u["t0"], B.CENTER) + nb[rng.integers(len(nb))].astype(float)
        rc = seg - np.median(seg, axis=1, keepdims=True)
        base = np.median(rc[:B.CENTER - 12], axis=0)
        x = rc - base
        z = _bp(rc) / fmad[None]
        k = B.CENTER + int(np.argmin(np.min(z[B.CENTER - 3:B.CENTER + 4], axis=1))) - 3
        ch = int(np.argmin(z[k]))
        cols["wave"].append(z[k - 15:k + 30].astype(np.float16))
        cols["amp3"].append((-(x[k - 1:k + 2].mean(axis=0))).astype(np.float32))
        cols["ch"].append(ch); cols["snr"].append(float(-z[k, ch])); cols["t"].append(float(t)); cols["unit"].append(int(j))
        cols["true_pos"].append(u["truth0"] + np.array([0.0, dd])); cols["drift"].append(float(dd))
    out = {k: np.asarray(v) for k, v in cols.items()}
    rcn = nb[:600].astype(float); rcn = rcn - np.median(rcn, axis=2, keepdims=True); rcn = rcn.reshape(-1, rcn.shape[2])
    out["E"] = E; out["sigma"] = (np.median(np.abs(rcn - np.median(rcn, axis=0)), axis=0) / 0.6745).astype(float); out["cells"] = [u["cell"] for u in units]
    return out


# ------------------------------------------------------------------------------------------------ trackers
def track_positions(t: np.ndarray, P: np.ndarray, method: str, **kw) -> np.ndarray:
    """Online position track of one unit from its single-event positions P [n, 2] at times t (causal)."""
    n = len(P); out = np.zeros_like(P)
    if method == "static":                     # running cumulative mean (no forgetting)
        out = np.cumsum(P, axis=0) / np.arange(1, n + 1)[:, None]
    elif method == "window":
        W = kw.get("W", 30)
        for i in range(n):
            out[i] = np.median(P[max(0, i - W + 1):i + 1], axis=0)
    elif method == "ema":
        a = kw.get("alpha", 0.05); m = P[0].copy()
        for i in range(n):
            m = P[i] if i == 0 else (1 - max(a, 1 / (i + 1))) * m + max(a, 1 / (i + 1)) * P[i]
            out[i] = m
    elif method == "kalman":
        q = kw.get("q", 0.5) ** 2; r = kw.get("r", 15.0) ** 2   # random-walk drift per second, measurement noise
        m = P[0].copy(); v = r; tp = t[0]
        for i in range(n):
            dt = max(t[i] - tp, 0.0); tp = t[i]
            v = v + q * dt
            g = v / (v + r)
            m = m + g * (P[i] - m); v = (1 - g) * v
            out[i] = m
    elif method == "kalman_cp":                # Kalman + CUSUM change-point reset on the axial innovation
        q = kw.get("q", 0.5) ** 2; r = kw.get("r", 15.0) ** 2; h = kw.get("h", 8.0); k_ = kw.get("k", 0.5)
        m = P[0].copy(); v = r; tp = t[0]; cp = cm = 0.0; buf = []
        for i in range(n):
            dt = max(t[i] - tp, 0.0); tp = t[i]
            v = v + q * dt
            z = (P[i, 1] - m[1]) / np.sqrt(v + r)
            cp = max(0.0, cp + z - k_); cm = max(0.0, cm - z - k_)
            buf = (buf + [P[i]])[-10:]
            if cp > h or cm > h:
                m = np.median(np.array(buf), axis=0); v = r / len(buf); cp = cm = 0.0
            else:
                g = v / (v + r); m = m + g * (P[i] - m); v = (1 - g) * v
            out[i] = m
    else:
        raise ValueError(method)
    return out
