"""Phase 14 morphology-disjoint biophysical source-count benchmark.

Templates: MEArec/NEURON (BBP L5 cell models). No NeuroSig forward model generates any event here.

  pool         cell models (morphologies)                 template files                       probe   noise
  p14dev       TTPC1, LBC, MC                             p11 + p12final sets                  NP1     IBL A-C (dev)
  p14select    UTPC, STPC, NBC                            p11 + p12final sets                  NP1     IBL E-H (select)
  p14sel_np2   UTPC, STPC, NBC                            templates_sel_np2 (fresh positions)  NP2     IBL E-H
  p14final     TTPC2, BP, BTC, ChC, DBC, NGC, SBC (never  templates_final_np1                  NP1     IBL T-W (fresh)
               simulated before Phase 14)
  p14final_np2 the same 7 unseen models                   templates_final_np2                  NP2     IBL T-W

Dev and select morphologies are disjoint, and both are disjoint from final. Real background noise (event-free raw AP
segments, sorter-free by >= 3 ms) is added in axial contact order; NP2 events reuse NP1 noise (stated limitation).

Event families (seeded per pool; the resolvable rule is the Phase 9 one, fixed before evaluation: min separation
>= 60 um, every source SNR >= 6, amplitude ratio >= 1/3):
  singles  clean | ampjit (+-30%) | artifact (2-sample impulse on 3 contacts) | dead (3 zeroed contacts) |
           corrnoise (one extra spatially smooth, temporally coloured noise source) | lowsnr (SNR 3-5)
  pairs    dt 0 | 1-3 | 4-9 | 10-20 samples at 40-140 um | near (20-60 um, dt 0-9) | unequal (ratio 0.2-0.5) |
           same (both sources from one template = identical shapes) | deadpair (3 dead contacts)
  triples  dt 1-9, 60-120 um
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from edge.types import Snippet

ROOT = Path(__file__).resolve().parents[2]
SEG_LEN, CENTER = 120, 50
N_PRE, N_POST = 40, 60
DEV_MODELS = ("L5_TTPC1_cADpyr232_1", "L5_LBC_bAC217_1", "L5_MC_bAC217_1")
SEL_MODELS = ("L5_UTPC_cADpyr232_1", "L5_STPC_cADpyr232_1", "L5_NBC_bAC217_1")
POOLS = {
    "p14dev": dict(files=("data/phase11/biophys/templates.h5", "data/phase12/biophys_final/templates.h5"),
                   models=DEV_MODELS, probe="Neuropixels-128", noise="dev", seed=20271441),
    "p14select": dict(files=("data/phase11/biophys/templates.h5", "data/phase12/biophys_final/templates.h5"),
                      models=SEL_MODELS, probe="Neuropixels-128", noise="select", seed=20271442),
    "p14sel_np2": dict(files=("data/phase14/templates_sel_np2.h5",), models=SEL_MODELS, probe="Neuropixels2-128",
                       noise="select", seed=20271443),
    "p14final": dict(files=("data/phase14/templates_final_np1.h5",), models=None, probe="Neuropixels-128",
                     noise="p14final", seed=20271444),
    "p14final_np2": dict(files=("data/phase14/templates_final_np2.h5",), models=None, probe="Neuropixels2-128",
                         noise="p14final", seed=20271445),
}
LATERAL = {"Neuropixels-128": 35.0, "Neuropixels2-128": 27.0}      # NP1 columns -> 11/27/43/59 um, NP2 -> 11/43 um
ROW = {"Neuropixels-128": 40.0, "Neuropixels2-128": 15.0}          # axial shift that maps the contact lattice onto itself


@dataclass
class TemplateSet:
    W: np.ndarray            # [n, C_all, T] at 30 kHz
    loc: np.ndarray          # [n, 3] MEArec (dist, lateral, axial)
    cell: np.ndarray
    E_all: np.ndarray        # [C_all, 2] probe plane (lateral + shift, axial)
    probe: str


@lru_cache(maxsize=8)
def template_set(pool: str) -> TemplateSet:
    import MEArec as mr
    import MEAutility as mu
    cfg = POOLS[pool]
    Ws, locs, cells = [], [], []
    for f in cfg["files"]:
        T = mr.load_templates(str(ROOT / f))
        c = np.array([str(x) for x in T.celltypes])
        keep = np.ones(len(c), bool) if cfg["models"] is None else np.isin(c, cfg["models"])
        Ws.append(resample_poly(np.asarray(T.templates, float)[keep], 15, 16, axis=2)); locs.append(np.asarray(T.locations, float)[keep]); cells.append(c[keep])
    P = mu.return_mea(cfg["probe"]).positions
    E = np.stack([P[:, 1] + LATERAL[cfg["probe"]], P[:, 2]], axis=1)
    return TemplateSet(np.concatenate(Ws), np.concatenate(locs), np.concatenate(cells), E, cfg["probe"])


def view(ts: TemplateSet, k: int, n_contacts: int = 32):
    """(W [T, 32] uV, E [32, 2], truth (lateral, axial), trough sample, peak contact) on the 32 contacts axially nearest
    the soma, ordered by (axial, lateral)."""
    sel = np.sort(np.argsort(np.abs(ts.E_all[:, 1] - ts.loc[k, 2]), kind="stable")[:n_contacts])
    sel = sel[np.lexsort((ts.E_all[sel, 0], ts.E_all[sel, 1]))]
    W = ts.W[k][sel].T
    t0, pk = np.unravel_index(int(np.argmin(W)), W.shape)
    return W, ts.E_all[sel].copy(), np.array([ts.loc[k, 1] + LATERAL[ts.probe], ts.loc[k, 2]]), int(t0), int(pk)


def shifted(ts: TemplateSet, k: int, E_view: np.ndarray, dz: float):
    """Template k with its soma moved axially by dz (a lattice multiple), sampled on E_view (zeros off the probe)."""
    W = np.zeros((ts.W.shape[2], len(E_view)))
    for c, (x, y) in enumerate(E_view):
        j = np.flatnonzero((np.abs(ts.E_all[:, 0] - x) < 0.5) & (np.abs(ts.E_all[:, 1] - (y - dz)) < 0.5))
        if len(j):
            W[:, c] = ts.W[k][j[0]]
    t0 = int(np.unravel_index(int(np.argmin(ts.W[k])), ts.W[k].shape)[1])
    return W, np.array([ts.loc[k, 1] + LATERAL[ts.probe], ts.loc[k, 2] + dz]), t0


# ------------------------------------------------------------------------------------------------ noise
def _noise_windows(name: str) -> list[dict]:
    if name in ("dev", "select"):
        from edge.phase12.biosim import _noise_windows as nw
        return nw(name)
    if name == "p14final":
        return json.loads((ROOT / "reports/phase14/split_manifest_phase14.json").read_text())["windows"]
    raise ValueError(name)


def noise_bank(name: str, n: int = 4000, seed: int = 0) -> np.ndarray:
    if name in ("dev", "select"):
        from edge.phase12.biosim import noise_bank as nb
        return nb(name)
    p = ROOT / f"data/phase14/noise_{name}.npy"
    if p.exists():
        return np.load(p)
    from scripts.phase5.data import load_window, window_path
    rng = np.random.default_rng(seed); segs = []
    wins = _noise_windows(name); per = n // len(wins) + 1
    for w in wins:
        q = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not q.exists():
            raise FileNotFoundError(q)
        d = load_window(q); v = np.asarray(d["voltages_uv"], float); fs = d["fs"]
        order = np.argsort(np.asarray(d["coordinates_um"])[:, 1], kind="stable")
        busy = np.zeros(v.shape[0], bool)
        for s in np.rint(np.asarray(d["sorter_times_s"]) * fs).astype(int):
            busy[max(s - 90, 0):s + 90] = True
        ok = [s for s in range(0, v.shape[0] - SEG_LEN, 7) if not busy[s:s + SEG_LEN].any()]
        for s in rng.choice(ok, min(per, len(ok)), replace=False):
            segs.append(v[s:s + SEG_LEN][:, order].astype(np.float32))
    out = np.stack(segs)[:n]; np.save(p, out)
    return out


def noise_sigma(name: str) -> np.ndarray:
    if name in ("dev", "select"):
        from edge.phase12.biosim import noise_sigma as ns
        return ns(name)
    p = ROOT / f"data/phase14/noise_sigma_{name}.npy"
    if p.exists():
        return np.load(p)
    from scripts.phase5.data import load_window, window_path
    from scripts.real_detector import frozen_real_detector, preprocess
    out = []
    for w in _noise_windows(name):
        d = load_window(window_path(w["pid"], w["t0_s"], w["duration_s"]))
        f, _ = preprocess(d["voltages_uv"], frozen_real_detector(d["fs"]))
        order = np.argsort(np.asarray(d["coordinates_um"])[:, 1], kind="stable")
        out.append((np.median(np.abs(f - np.median(f, axis=0)), axis=0) / 0.6745)[order])
    sig = np.median(np.stack(out), axis=0); np.save(p, sig)
    return sig


# ------------------------------------------------------------------------------------------------ events
@dataclass
class Event:
    pool: str
    kind: str                # single | pair | triple
    cond: str
    sources: np.ndarray      # [K, 2] soma (lateral, axial)
    snr: tuple
    dt: int
    sep: float
    resolvable: bool
    snippet: Snippet
    tids: tuple
    cells: tuple


def place(W, t0, at):
    out = np.zeros((SEG_LEN, W.shape[1]))
    lo = at - t0
    a, b = max(lo, 0), min(lo + W.shape[0], SEG_LEN)
    if b > a:
        out[a:b] = W[a - lo:b - lo]
    return out


def _snip(i, seg, E, ref):
    E3 = np.column_stack([E, np.zeros(len(E))])
    return Snippet(i, N_PRE, ref - N_PRE, 30000.0, tuple(range(len(E))), seg[ref - N_PRE:ref + N_POST], E3, None, None)


def _corr_noise(rng, E, sig):
    """One extra noise source: spatially smooth (Gaussian, 60 um) and temporally coloured, 2-4 sigma at its centre."""
    c = E[rng.integers(len(E))]
    sp = np.exp(-np.sum((E - c) ** 2, axis=1) / (2 * 60.0 ** 2))
    t = np.convolve(rng.standard_normal(SEG_LEN + 20), np.hanning(9), "same")[10:10 + SEG_LEN]
    t = t / np.max(np.abs(t))
    return np.outer(t, sp * sig * rng.uniform(2, 4))


CAL_TEMPLATES = {"p14dev": "p14dev", "p14select": "p14dev", "p14final": "p14dev", "p14sel_np2": "p14sel_np2",
                 "p14final_np2": "p14sel_np2"}


def generate(pool: str, n_single: int = 150, n_pair: int = 60, n_triple: int = 60, calibration: bool = False,
             noise_scale: float = 1.0, noise_override=None) -> list[Event]:
    """n_single per single condition, n_pair per pair condition, n_triple triples.

    calibration=True: label-free threshold calibration set for ``pool`` = singles only, built from DEVELOPMENT
    morphologies (CAL_TEMPLATES; NP2 pools use the select-morphology NP2 set, never final morphologies) placed in the
    pool's own background noise, with a distinct seed. Only the target recording's event-free background is used."""
    cfg = POOLS[pool]
    ts = template_set(CAL_TEMPLATES[pool] if calibration else pool)
    nb, sig = noise_override if noise_override is not None else (noise_bank(cfg["noise"]), noise_sigma(cfg["noise"]))
    nb = nb * noise_scale
    rng = np.random.default_rng(cfg["seed"] + (1000 if calibration else 0))
    if calibration:
        n_pair = n_triple = 0
    ids = np.arange(len(ts.W)); out: list[Event] = []
    row = ROW[ts.probe]

    def scale(W, t0, pk, snr):
        return W * (snr * sig[pk] / max(-W[t0, pk], 1e-9))

    for cond in ("clean", "ampjit", "artifact", "dead", "corrnoise", "lowsnr"):
        for _ in range(n_single):
            k = int(rng.choice(ids)); snr = float(rng.uniform(3, 5) if cond == "lowsnr" else rng.uniform(5, 20))
            W, E, tr, t0, pk = view(ts, k)
            s = scale(W, t0, pk, snr) * (1 + (0.3 * rng.uniform(-1, 1) if cond == "ampjit" else 0.0))
            seg = place(s, t0, CENTER) + nb[rng.integers(len(nb))].astype(float)
            if cond == "artifact":
                c = rng.choice(32, 3, replace=False); t = int(rng.integers(CENTER - 15, CENTER + 25))
                seg[t:t + 2, c] += rng.choice([-1, 1]) * 6 * sig[c]
            if cond == "corrnoise":
                seg += _corr_noise(rng, E, sig)
            if cond == "dead":
                seg[:, rng.choice(32, 3, replace=False)] = 0.0
            ref = CENTER + int(rng.integers(-2, 3))
            out.append(Event(pool, "single", cond, tr[None], (snr,), 0, 0.0, False, _snip(len(out), seg, E, ref), (k,), (ts.cell[k],)))

    def multi(K, dt_lo, dt_hi, sep_lo, sep_hi, cond, ratio=(0.5, 1.0), same=False, dead=0):
        k1 = int(rng.choice(ids))
        W1, E, tr1, t01, pk1 = view(ts, k1)
        snr1 = float(rng.uniform(6, 20))
        seg = place(scale(W1, t01, pk1, snr1), t01, CENTER)
        truths, snrs, ts_, ks = [tr1], [snr1], [CENTER], [k1]
        for j in range(1, K):
            k2 = k1 if same else int(rng.choice(ids[ids != k1]))
            side = float(rng.choice([-1, 1])) if j == 1 else (-np.sign(truths[1][1] - tr1[1]) or 1.0)
            target = tr1[1] + side * rng.uniform(sep_lo, sep_hi)
            dz = round((target - ts.loc[k2, 2]) / row) * row
            if same and dz == 0:
                return None
            W2, tr2, t02 = shifted(ts, k2, E, dz)
            if not np.any(W2):
                return None
            pk2 = int(np.argmin(W2[t02]))
            if -W2[t02, pk2] <= 1e-6:
                return None
            snr2 = snr1 * float(rng.uniform(*ratio))
            dt = int(rng.integers(dt_lo, dt_hi + 1)) * int(rng.choice([-1, 1]))
            seg = seg + place(scale(W2, t02, pk2, snr2), t02, CENTER + dt)
            truths.append(tr2); snrs.append(snr2); ts_.append(CENTER + dt); ks.append(k2)
        seg = seg + nb[rng.integers(len(nb))].astype(float)
        if dead:
            seg[:, rng.choice(32, dead, replace=False)] = 0.0
        T = np.array(truths)
        sep = min(float(np.linalg.norm(T[a] - T[b])) for a in range(K) for b in range(a + 1, K))
        res = bool(sep >= 60 and min(snrs) >= 6 and min(snrs) / max(snrs) >= 1 / 3)
        ref = int(ts_[int(np.argmax(snrs))]) + int(rng.integers(-2, 3))
        return Event(pool, "pair" if K == 2 else "triple", cond, T, tuple(snrs), abs(ts_[1] - ts_[0]), sep, res,
                     _snip(len(out), seg, E, ref), tuple(ks), tuple(ts.cell[k] for k in ks))

    specs = [(2, 0, 0, 40, 140, "dt0", {}), (2, 1, 3, 40, 140, "dt1-3", {}), (2, 4, 9, 40, 140, "dt4-9", {}),
             (2, 10, 20, 40, 140, "dt10-20", {}), (2, 0, 9, 20, 60, "near", {}),
             (2, 0, 9, 40, 140, "unequal", {"ratio": (0.2, 0.5)}), (2, 0, 9, 40, 140, "same", {"same": True}),
             (2, 0, 9, 40, 140, "deadpair", {"dead": 3})]
    for K, lo, hi, s0, s1, cond, kw in specs:
        n = 0
        while n < n_pair:
            e = multi(K, lo, hi, s0, s1, cond, **kw)
            if e is not None:
                out.append(e); n += 1
    n = 0
    while n < n_triple:
        e = multi(3, 1, 9, 60, 120, "triple")
        if e is not None:
            out.append(e); n += 1
    return out
