"""Phase 12.2: unit-level localization by pooling repeated spikes.

Event-level localization asks where one detected event came from; unit-level localization asks where the neuron that
produced a set of events lies. This module collects every sorter event of a session (the provisional grouping labels,
not ground truth), keeps a raw segment around it, and compares six estimators on disjoint pools of N spikes:

  single    each event localized alone (N = 1 is the matched event-level baseline)
  mean      mean raw multichannel waveform of the pool, localized once
  median    element-wise median waveform
  robust    Huber-reweighted mean waveform (weights from each spike's residual to the current template)
  posagg    SNR^2-weighted geometric median of the single-event positions (confidence-weighted aggregation)
  joint     one line-source position fitted jointly to all spikes' trough-amplitude vectors, a free non-negative
            amplitude per spike (variable projection)

Two engine families are used so that the pooling effect is not confounded with the engine:
  ws   the frozen P11 weighted_scan engine applied to a raw snippet (the Phase 11 headline engine)
  l20  the Phase 11 line20 amplitude fit on E3-aligned trough amplitudes (the 11B baseline model)
Pools are formed after a fixed seeded shuffle, are disjoint, and never reuse an event. Stability is the median
pairwise distance between estimates from disjoint pools of the same unit; drift is the distance between the earliest
and latest temporal pools compared with random pools of the same size.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from edge.phase11 import forward_models as F
from edge.phase11.referencing import align_sample_raw, event_quality, load_events, match_ab, raw_car
from edge.snippets import extract_snippet
from edge.types import Snippet
from scripts.phase5.data import load_window, window_path

SEG = (-50, 70)                 # stored raw segment around the sorter time (samples)
N_PRE, N_POST = 40, 60          # snippet geometry used by the P9-P11 engines
NS = (1, 5, 10, 25, 50, 100)
MAX_POOLS = 8
MAX_SINGLES = 40
WAVE = (-20, 40)


@dataclass
class SessionEvents:
    session: str
    array: object
    E: np.ndarray
    fs: float
    seg: np.ndarray            # [n, 120, C] raw segments (float32) around the sorter time
    sigma_rc: np.ndarray       # [n_windows, C] MAD of raw - median CAR per source window
    shift: np.ndarray          # E3 aligned sample - sorter sample
    t_s: np.ndarray            # absolute probe time (window t0 + sample / fs)
    unit: np.ndarray
    sorter_ch: np.ndarray
    snr: np.ndarray            # at the aligned sample, frozen preprocessing
    matched: np.ndarray        # has a detector-v2.0 partner (11A rule)
    win: np.ndarray
    extra: dict = field(default_factory=dict)

    def snippet(self, i: int, aligned: bool = True, v: np.ndarray | None = None) -> Snippet:
        """Snippet for event i (or for a pooled waveform ``v`` cut on the aligned grid)."""
        if v is None:
            c = -SEG[0] + (int(self.shift[i]) if aligned else 0)
            v = self.seg[i, c - N_PRE:c + N_POST].astype(float)
        return Snippet(int(i), N_PRE, 0, self.fs, tuple(range(v.shape[1])), v, self.E, int(self.sorter_ch[i]) if i >= 0 else None, None)

    def aligned_wave(self, idx) -> np.ndarray:
        """[n, N_PRE+N_POST, C] aligned raw snippets."""
        c = -SEG[0] + self.shift[idx]
        return np.stack([self.seg[i, k - N_PRE:k + N_POST] for i, k in zip(idx, c)]).astype(float)

    def amps(self, idx) -> np.ndarray:
        """E3 trough-amplitude vectors (positive = trough depth): -(rc[s] - median(rc[s-40:s-12]))."""
        c = -SEG[0] + self.shift[idx]
        out = []
        for i, k in zip(idx, c):
            seg = self.seg[i].astype(float)
            rc = seg - np.median(seg, axis=1, keepdims=True)          # the engines' raw - median CAR signal
            base = np.median(rc[k - 40:k - 12], axis=0)
            out.append(-(rc[k] - base))
        return np.array(out, float)


def collect_session(wins: list[dict], session: str, max_per_unit: int = 220, seed: int = 0, log=None) -> SessionEvents | None:
    """All sorter events of one session's windows (capped per unit by a seeded random choice, kept in time order)."""
    rng = np.random.default_rng(seed)
    cols = defaultdict(list)
    array = E = fs = None
    for wi, w in enumerate(wins):
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        we = load_events(w, load_window(p))
        array, E, fs = we.array, we.array.xyz_um(), we.fs
        _, sig = raw_car(we)
        matched = {r for _, r, _ in match_ab(we, we.det_sample)}
        for i, (s, ch, u) in enumerate(zip(we.sorter_sample, we.sorter_channel, we.sorter_cluster)):
            s, ch = int(s), int(ch)
            if s + SEG[0] - 2 < 0 or s + SEG[1] + 2 > we.raw.shape[0]:
                continue
            s3 = align_sample_raw(we, s, ch, 3)
            q = event_quality(we, s3, ch)
            cols["seg"].append(we.raw[s + SEG[0]:s + SEG[1]].astype(np.float32))
            cols["shift"].append(s3 - s); cols["t"].append(float(w["t0_s"]) + s / fs); cols["unit"].append(int(u))
            cols["ch"].append(ch); cols["snr"].append(q["snr"]); cols["matched"].append(i in matched); cols["win"].append(wi)
        cols["sigw"].append(sig.astype(np.float32))
        if log:
            log(f"{session} window {wi + 1}/{len(wins)} events {len(cols['t'])}")
    if not cols["t"]:
        return None
    unit = np.array(cols["unit"]); t = np.array(cols["t"])
    keep = []
    for u in np.unique(unit):
        idx = np.flatnonzero(unit == u)
        if len(idx) > max_per_unit:
            idx = np.sort(rng.choice(idx, max_per_unit, replace=False))
        keep += list(idx)
    keep = np.array(sorted(keep, key=lambda i: t[i]))
    A = lambda k, dt=None: np.asarray(cols[k], dt)[keep] if k != "seg" else np.stack([cols[k][i] for i in keep])
    return SessionEvents(session, array, E, fs, A("seg"), np.stack(cols["sigw"]), A("shift", int), A("t", float), A("unit", int),
                         A("ch", int), A("snr", float), A("matched", bool), A("win", int),
                         {"n_total_before_cap": int(len(unit))})


# ------------------------------------------------------------------------------------------------ pooled estimators
def huber_mean(X: np.ndarray, iters: int = 3, c: float = 1.5) -> np.ndarray:
    """Huber-reweighted mean over the first axis (residual = Frobenius distance to the current template)."""
    T = np.median(X, axis=0)
    for _ in range(iters):
        r = np.sqrt(np.sum((X - T[None]) ** 2, axis=tuple(range(1, X.ndim))))
        s = max(float(np.median(r)), 1e-9)
        w = np.minimum(1.0, c * s / np.maximum(r, 1e-12))
        T = np.tensordot(w, X, axes=1) / w.sum()
    return T


def weighted_geomedian(P: np.ndarray, w: np.ndarray, iters: int = 50) -> np.ndarray:
    m = np.average(P, axis=0, weights=w)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(P - m, axis=1), 1e-6)
        m2 = np.average(P, axis=0, weights=w / d)
        if np.linalg.norm(m2 - m) < 1e-3:
            return m2
        m = m2
    return m


def joint_line20(A: np.ndarray, sigma: np.ndarray, E: np.ndarray, cache: dict | None = None, y_margin=40.0,
                 x_range=(-20.0, 80.0)) -> np.ndarray:
    """One line20 position for n spikes with per-spike non-negative amplitudes: argmin sum_k ||a_k - A_k g(theta)||_W^2."""
    w = 1.0 / np.square(sigma)
    key = ("joint", E.tobytes())
    if cache is not None and key in cache:
        P, G = cache[key]
    else:
        P = F._grid(E, y_margin, x_range); G = F._grid_gains(F.Model("line20"), P, E)
        if cache is not None:
            cache[key] = (P, G)
    num = G @ (A * w[None]).T                       # [P, n]
    den = (G * G) @ w                               # [P]
    amp = np.maximum(num / np.maximum(den, 1e-18)[:, None], 0.0)
    score = np.sum(amp * num, axis=1)               # = sum_k tot_k - sse_k up to constants
    th0 = P[int(np.argmax(score))]
    sw = np.sqrt(w)

    def resid(th):
        g = F.line_gain(th[0], th[1], E, 20.0)
        gw = g * sw
        a = np.maximum((A * sw[None]) @ gw / max(gw @ gw, 1e-18), 0.0)
        return ((A * sw[None]) - a[:, None] * gw[None]).ravel()
    lo = np.array([x_range[0], E[:, 1].min() - y_margin]); hi = np.array([x_range[1], E[:, 1].max() + y_margin])
    sol = least_squares(resid, np.clip(th0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", x_scale=10.0, max_nfev=40)
    return sol.x


class UnitLocalizer:
    """All estimators for one session (engines built once per session array)."""

    def __init__(self, se: SessionEvents, ws_engine):
        self.se, self.ws, self.cache = se, ws_engine, {}
        self.sigma = np.median(se.sigma_rc, axis=0).astype(float)

    def _ws(self, v: np.ndarray, i_ref: int):
        r = self.ws.estimate(self.se.snippet(i_ref, v=v))
        return np.array(r.xyz_um[:2]) if (r.available and r.xyz_um is not None) else None

    def _l20(self, a: np.ndarray):
        f = F.fit(F.Model("line20"), a, self.sigma, self.se.E, cache=self.cache)
        return f.theta[:2].copy() if f.amps[0] > 0 else None

    def singles(self, idx, aligned=True):
        """Per-event ws and l20 positions (None when unavailable)."""
        W = self.se.aligned_wave(idx) if aligned else None
        A = self.se.amps(idx)
        out = []
        for j, i in enumerate(idx):
            v = W[j] if aligned else None
            ws = self._ws(v, i) if aligned else self._ws(self.se.snippet(i, aligned=False).voltages_uv, i)
            out.append((ws, self._l20(A[j])))
        return out

    def pooled(self, idx, single_cache: dict) -> dict:
        idx = np.asarray(idx)
        W = self.se.aligned_wave(idx); A = self.se.amps(idx); i0 = int(idx[0])
        res = {"ws_mean": self._ws(W.mean(0), i0), "ws_median": self._ws(np.median(W, 0), i0),
               "ws_robust": self._ws(huber_mean(W), i0),
               "l20_mean": self._l20(A.mean(0)), "l20_median": self._l20(np.median(A, 0)), "l20_robust": self._l20(huber_mean(A)),
               "l20_joint": joint_line20(A, self.sigma, self.se.E, self.cache)}
        w = np.minimum(self.se.snr[idx], 30.0) ** 2
        for fam, k in (("ws", 0), ("l20", 1)):
            P = [(single_cache[int(i)][k], wi) for i, wi in zip(idx, w) if single_cache[int(i)][k] is not None]
            res[f"{fam}_posagg"] = weighted_geomedian(np.array([p for p, _ in P]), np.array([x for _, x in P])) if P else None
        return res


METHODS = ("single", "mean", "median", "robust", "posagg", "joint")


def _pairwise(P: list) -> float | None:
    P = [p for p in P if p is not None]
    if len(P) < 2:
        return None
    P = np.array(P)
    d = np.linalg.norm(P[:, None] - P[None], axis=2)[np.triu_indices(len(P), 1)]
    return float(np.median(d))


def unit_stability(ul: UnitLocalizer, units: dict, seed: int = 1, ns=NS) -> list[dict]:
    """``units``: {unit: event indices (time ordered)}. Returns one row per (unit, N) with the pairwise disagreement of
    each estimator across disjoint pools, the temporal (first-vs-last pool) distance and pool counts."""
    se = ul.se
    rng = np.random.default_rng(seed)
    rows = []
    for u, idx in units.items():
        idx = np.asarray(idx)
        single = {int(i): s for i, s in zip(idx, ul.singles(idx))}
        perm = idx[rng.permutation(len(idx))]
        for N in ns:
            M = len(idx) // N
            if M < 2:
                continue
            row = {"session": se.session, "unit": int(u), "N": N, "n_events": int(len(idx)),
                   "snr_median": float(np.median(se.snr[idx]))}
            if N == 1:
                pick = perm[:MAX_SINGLES]
                row["ws_single_unaligned"] = _pairwise([ul._ws(se.snippet(int(i), aligned=False).voltages_uv, int(i)) for i in pick])
                for fam, k in (("ws", 0), ("l20", 1)):
                    P = [single[int(i)][k] for i in pick]
                    for m in METHODS:
                        if m == "joint" and fam == "ws":
                            continue
                        row[f"{fam}_{m}"] = _pairwise(P)
                    allp = [single[int(i)][k] for i in idx if single[int(i)][k] is not None]
                    row[f"{fam}_valid"] = len(allp) / len(idx)
                    if len(allp) >= 2:      # temporal: earliest vs latest event
                        row[f"{fam}_first_last"] = float(np.linalg.norm(allp[0] - allp[-1]))
                rows.append(row); continue
            pools = [perm[k * N:(k + 1) * N] for k in range(min(M, MAX_POOLS))]
            est = [ul.pooled(p, single) for p in pools]
            for key in est[0]:
                row[key] = _pairwise([e[key] for e in est])
            for fam in ("ws", "l20"):
                row[f"{fam}_single"] = _pairwise([single[int(i)][0 if fam == "ws" else 1] for i in perm[:MAX_SINGLES]])
            # temporal drift: first and last contiguous pools in time (mean estimator) vs random-pool disagreement
            first, last = ul.pooled(idx[:N], single), ul.pooled(idx[-N:], single)
            for key in ("ws_mean", "l20_mean", "l20_joint"):
                if first[key] is not None and last[key] is not None:
                    row[f"{key}_first_last"] = float(np.linalg.norm(first[key] - last[key]))
            row["span_s"] = float(se.t_s[idx[-1]] - se.t_s[idx[0]])
            row["n_pools"] = len(pools)
            rows.append(row)
    return rows


def unit_position(ul: UnitLocalizer, idx) -> dict:
    """Unit-level position from all given events (mean waveform, ws and l20) for proxy-bias analysis."""
    p = ul.pooled(np.asarray(idx), {int(i): s for i, s in zip(idx, ul.singles(np.asarray(idx)))})
    return {k: (None if v is None else [float(v[0]), float(v[1])]) for k, v in p.items()}


# ------------------------------------------------------------------------------------------------ cache + summaries
def save_session(se: SessionEvents, path) -> None:
    np.savez(path, seg=se.seg, sigma_rc=se.sigma_rc, shift=se.shift, t_s=se.t_s, unit=se.unit, sorter_ch=se.sorter_ch,
             snr=se.snr, matched=se.matched, win=se.win, E=se.E, fs=se.fs, session=se.session,
             n_total=se.extra.get("n_total_before_cap", -1))


def load_session(path, array) -> SessionEvents:
    with np.load(path, allow_pickle=False) as z:
        return SessionEvents(str(z["session"]), array, z["E"], float(z["fs"]), z["seg"], z["sigma_rc"], z["shift"], z["t_s"],
                             z["unit"], z["sorter_ch"], z["snr"], z["matched"], z["win"], {"n_total_before_cap": int(z["n_total"])})


def _boot_rel(a: np.ndarray, b: np.ndarray, seed=0, n=2000):
    """Relative change of the median over units, b vs a (paired units), with a unit-bootstrap 95% interval."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a)))
    r = np.median(b[idx], axis=1) / np.maximum(np.median(a[idx], axis=1), 1e-9) - 1
    return {"n_units": int(len(a)), "rel_change": float(np.median(b) / max(np.median(a), 1e-9) - 1),
            "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def summarize_stability(rows: list[dict], keys: list[str], fixed_min_events: int = 100) -> dict:
    """Per N and estimator: median / p90 over units of the pooled disagreement, relative change against the same
    units' single-event disagreement (same engine family), and the same on a fixed unit set (>= fixed_min_events)."""
    out = {}
    fixed_units = {(r["session"], r["unit"]) for r in rows if r["N"] == 1 and r["n_events"] >= fixed_min_events}
    for N in sorted({r["N"] for r in rows}):
        R = [r for r in rows if r["N"] == N]
        o = {"n_units": len(R)}
        for k in keys:
            v = np.array([np.nan if r.get(k) is None else r[k] for r in R], float)
            if not np.isfinite(v).any():
                continue
            fam = k.split("_")[0]
            base = np.array([np.nan if r.get(f"{fam}_single") is None else r[f"{fam}_single"] for r in R], float)
            fx = np.array([(r["session"], r["unit"]) in fixed_units for r in R])
            o[k] = {"median": float(np.nanmedian(v)), "p90": float(np.nanpercentile(v, 90)),
                    "vs_single": _boot_rel(base, v, seed=N), "fixed_set": {"n_units": int(fx.sum()),
                    "median": float(np.nanmedian(v[fx])) if fx.any() and np.isfinite(v[fx]).any() else None,
                    "vs_single": _boot_rel(base[fx], v[fx], seed=N + 1)}}
            fl = np.array([np.nan if r.get(f"{k}_first_last") is None else r[f"{k}_first_last"] for r in R], float)
            if np.isfinite(fl).any():
                o[k]["first_last_median"] = float(np.nanmedian(fl))
                o[k]["first_last_over_random"] = float(np.nanmedian(fl) / max(np.nanmedian(v), 1e-9))
        out[str(N)] = o
    return out
