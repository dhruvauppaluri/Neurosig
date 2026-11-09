"""Phase 12 model-mismatched simulator: MEArec/NEURON biophysical templates + real IBL background noise.

Nothing here uses a NeuroSig forward model. Templates (data/phase11/biophys/templates.h5: 72 templates, 6 BBP L5 cell
types x 12, Neuropixels-128 contact layout, seed 20271108) are resampled 32 -> 30 kHz. For each template the 32
contacts nearest the soma axially are used; probe-plane mapping lateral = MEArec y, axial = MEArec z (11B
convention). Background noise is real: raw AP segments from cached IBL windows at times >= 3 ms away from any sorter
spike (event-free by the sorter's account), added channel-by-channel in axial order. Real noise is spatially
correlated and non-Gaussian; the template geometry is not the noise probe's, which is stated as a limitation.

Template pools are disjoint and stratified by cell type: 'dev' = even index within each cell type, 'select' = odd.
Noise pools come from disjoint sessions: 'dev' = sessions A-C (Phase 5 dev windows), 'select' = sessions E-H.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
SEG_LEN = 120
CENTER = 50
LATERAL_SHIFT = 35.0        # MEArec columns -24/-8/8/24 um -> 11/27/43/59 um, the Neuropixels 1.0 lateral positions


TEMPLATE_FILES = {"p11": "data/phase11/biophys/templates.h5", "final": "data/phase12/biophys_final/templates.h5"}


def tset(pool: str) -> str:
    """Template set for a pool: dev/select split the Phase 11 set; final uses the fresh Phase 12 set."""
    return "final" if pool == "final" else "p11"


@lru_cache(maxsize=2)
def templates(which: str = "p11"):
    import MEArec as mr
    import MEAutility as mu
    T = mr.load_templates(str(ROOT / TEMPLATE_FILES[which]))
    P = mu.return_mea("Neuropixels-128").positions
    E_all = np.stack([P[:, 1], P[:, 2], np.zeros(len(P))], axis=1)
    tm = resample_poly(np.asarray(T.templates, float), 15, 16, axis=2)          # 32 -> 30 kHz
    cells = np.array([str(c) for c in T.celltypes])
    return tm, np.asarray(T.locations, float), cells, E_all


def template_pool(pool: str) -> list[int]:
    _, _, cells, _ = templates(tset(pool))
    if pool == "final":
        return list(range(len(cells)))
    out = []
    for c in sorted(set(cells)):
        idx = np.flatnonzero(cells == c)
        out += list(idx[0::2] if pool == "dev" else idx[1::2])
    return sorted(out)


def template_view(k: int, which: str = "p11"):
    """(W [T, 32] uV at 30 kHz, E [32, 3] probe-plane, truth [2] (lateral, axial), trough sample, peak contact)."""
    tm, loc, _, E_all = templates(which)
    sel = np.sort(np.argsort(np.abs(E_all[:, 1] - loc[k, 2]))[:32])
    sel = sel[np.argsort(E_all[sel, 1] * 1000 + E_all[sel, 0])]
    W = tm[k][sel].T
    t0, pk = np.unravel_index(int(np.argmin(W)), W.shape)
    E = E_all[sel].copy(); E[:, 0] += LATERAL_SHIFT
    return W, E, np.array([loc[k, 1] + LATERAL_SHIFT, loc[k, 2]]), int(t0), int(pk)


def _noise_windows(pool: str) -> list[dict]:
    from edge.phase9 import harness as H
    if pool == "dev":
        return H.manifest_windows("dev")
    if pool == "select":
        return H.manifest_windows("locked")
    from edge.phase12.splits import split_windows
    return split_windows("final")


def noise_bank(pool: str, n: int = 4000, seed: int = 0) -> np.ndarray:
    """[n, SEG_LEN, 32] raw event-free segments (cached under data/phase12)."""
    p = ROOT / f"data/phase12/noise_{pool}.npy"
    if p.exists():
        return np.load(p)
    from edge.phase9 import harness as H
    from scripts.phase5.data import load_window, window_path
    wins = _noise_windows(pool)
    rng = np.random.default_rng(seed)
    segs = []
    per = max(n // len(wins) + 1, 1)
    for w in wins:
        q = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not q.exists():
            continue
        d = load_window(q); v = np.asarray(d["voltages_uv"], float); fs = d["fs"]
        y = np.asarray(d["coordinates_um"])[:, 1]; order = np.argsort(y, kind="stable")
        busy = np.zeros(v.shape[0], bool)
        for s in np.rint(np.asarray(d["sorter_times_s"]) * fs).astype(int):
            busy[max(s - 90, 0):s + 90] = True
        ok = [s for s in range(0, v.shape[0] - SEG_LEN, 7) if not busy[s:s + SEG_LEN].any()]
        if not ok:
            continue
        for s in rng.choice(ok, min(per, len(ok)), replace=False):
            segs.append(v[s:s + SEG_LEN][:, order].astype(np.float32))
    out = np.stack(segs)[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, out)
    return out


def noise_sigma(pool: str) -> np.ndarray:
    """Per-contact (axial order) MAD noise of the frozen real-detector preprocessing (300-6000 Hz, median CAR) of the
    noise-source windows: the same noise measure as real-event SNR in 11A. SNR in this simulator is relative to it."""
    p = ROOT / f"data/phase12/noise_sigma_{pool}.npy"
    if p.exists():
        return np.load(p)
    from edge.phase9 import harness as H
    from scripts.phase5.data import load_window, window_path
    from scripts.real_detector import frozen_real_detector, preprocess
    wins = _noise_windows(pool)
    out = []
    for w in wins:
        q = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not q.exists():
            continue
        d = load_window(q); f, _ = preprocess(d["voltages_uv"], frozen_real_detector(d["fs"]))
        order = np.argsort(np.asarray(d["coordinates_um"])[:, 1], kind="stable")
        out.append((np.median(np.abs(f - np.median(f, axis=0)), axis=0) / 0.6745)[order])
    sig = np.median(np.stack(out), axis=0)
    np.save(p, sig)
    return sig


def noise_mad(seg_bank: np.ndarray) -> np.ndarray:
    x = seg_bank - np.median(seg_bank, axis=2, keepdims=True)          # raw - median CAR
    x = x.reshape(-1, x.shape[-1])
    return np.median(np.abs(x - np.median(x, axis=0)), axis=0) / 0.6745


@dataclass
class SimEvent:
    seg: np.ndarray          # [SEG_LEN, 32] raw (template(s) + real noise)
    E: np.ndarray
    truth: np.ndarray        # [K, 2] (lateral, axial) per source
    template: tuple
    snr: tuple               # requested peak-trough SNR per source
    ref_sample: int          # nominal (jittered) reference sample of source 0
    true_sample: tuple       # true trough sample per source
    kind: str = "single"


def place(W: np.ndarray, t0: int, at: int) -> np.ndarray:
    out = np.zeros((SEG_LEN, W.shape[1]))
    lo = at - t0
    a, b = max(lo, 0), min(lo + W.shape[0], SEG_LEN)
    out[a:b] = W[a - lo:b - lo]
    return out


def make_single(k: int, snr: float, noise: np.ndarray, sig: np.ndarray, rng, jitter: int = 2, amp_jitter: float = 0.0,
                artifact: bool = False, dead: int = 0, which: str = "p11") -> SimEvent:
    """One biophysical event at peak-trough SNR ``snr`` (relative to the real noise MAD at its peak contact)."""
    W, E, truth, t0, pk = template_view(k, which)
    scale = snr * sig[pk] / max(-W[t0, pk], 1e-9) * (1.0 + amp_jitter * rng.uniform(-1, 1))
    seg = place(W * scale, t0, CENTER) + noise[rng.integers(len(noise))].astype(float)
    if artifact:                                   # brief common-mode step / impulse on a few contacts
        c = rng.choice(32, 3, replace=False); t = int(rng.integers(CENTER - 15, CENTER + 25))
        seg[t:t + 2, c] += rng.choice([-1, 1]) * 6 * sig[c]
    if dead:
        seg[:, rng.choice(32, dead, replace=False)] = 0.0
    j = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    return SimEvent(seg, E, truth[None], (k,), (snr,), CENTER + j, (CENTER,))


def shifted_on_view(k: int, E_view: np.ndarray, dz: float, which: str = "p11") -> tuple[np.ndarray, np.ndarray, int]:
    """Template k with its soma moved axially by dz (a multiple of 40 um keeps the checkerboard layout exact), sampled
    on the probe-plane contacts E_view. Contacts that fall off the simulated probe read zero.
    Returns (W [T, 32], truth [2], trough sample)."""
    tm, loc, _, E_all = templates(which)
    lat = E_all[:, 0] + LATERAL_SHIFT
    W = np.zeros((tm.shape[2], len(E_view)))
    for c, (x, y) in enumerate(E_view[:, :2]):
        j = np.flatnonzero((np.abs(lat - x) < 0.5) & (np.abs(E_all[:, 1] - (y - dz)) < 0.5))
        if len(j):
            W[:, c] = tm[k][j[0]]
    t0 = int(np.unravel_index(int(np.argmin(tm[k])), tm[k].shape)[1])
    return W, np.array([loc[k, 1] + LATERAL_SHIFT, loc[k, 2] + dz]), t0
