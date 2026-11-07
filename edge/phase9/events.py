"""Corrected real-data event extraction shared by Phase 9 experiments.

Timebases: raw voltages are indexed by probe sample; cached windows (sync_version >= 3) store sorter times on the
probe clock relative to the window start. ``clock="session"`` reproduces the historical error (session-clock times
used as probe time) for control comparisons only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge import ibl
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray
from scripts.phase5.data import load_window, window_path

FS_DEFAULT = 30000.0
SYNC_OK = 3


def window_sorter_seconds(w: dict, clock: str = "probe") -> np.ndarray:
    """Sorter spike times in seconds from the window start on the requested clock assumption."""
    t = np.asarray(w["sorter_times_s"], float)
    if w.get("sync_version", 1) < SYNC_OK:
        raise ValueError("window sorter times are not on the probe clock; run scripts.phase5.data.resync_window")
    if clock == "probe":
        return t
    if clock == "session":     # historical bug: session-clock time treated as probe time
        return t + float(w["sync_offset_s"])
    raise ValueError(clock)


def window_array(w: dict) -> ElectrodeArray:
    sel = np.asarray(w["selected"], int)
    loc = np.full((int(sel.max()) + 1, 2), np.nan)
    loc[sel] = w["coordinates_um"][:, :2]
    return ibl.probe_geometry(sel, loc)


def sorter_snippets(w: dict, clock: str = "probe", max_events: int | None = None, seed: int = 0,
                    voltages: np.ndarray | None = None, n_pre: int | None = None, n_post: int | None = None):
    """Snippets centred on sorter spike samples. Returns (array, [snippet], meta list)."""
    fs = float(w["fs"])
    v = w["voltages_uv"] if voltages is None else voltages
    t = window_sorter_seconds(w, clock)
    idx = np.arange(len(t))
    if max_events is not None and len(idx) > max_events:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_events, replace=False))
    array = window_array(w)
    sn, meta = [], []
    for i in idx:
        s = int(round(t[i] * fs))
        if s - 40 < 0 or s + 60 >= v.shape[0]:
            continue
        kw = {}
        if n_pre is not None:
            kw["n_pre"] = n_pre
        if n_post is not None:
            kw["n_post"] = n_post
        ch = int(w["sorter_local_channel"][i])
        sn.append(extract_snippet(v, s, array, event_id=int(i), sample_rate_hz=fs, peak_channel=ch, **kw))
        meta.append({"cluster": int(w["sorter_clusters"][i]), "local_channel": ch, "sample": s,
                     "ref_y_um": float(array.xyz_um()[ch, 1]), "t0_s": float(w["t0_s"]), "pid": str(w["pid"])})
    return array, sn, meta


def trough_offset_ms(filtered: np.ndarray, samples: np.ndarray, channels: np.ndarray, fs: float,
                     half_window_ms: float = 3.0) -> np.ndarray:
    """Signed offset (ms) from the supplied sample to the filtered-trace minimum on the sorter peak channel."""
    h = int(round(half_window_ms * fs / 1000.0))
    out = np.full(len(samples), np.nan)
    for k, (s, c) in enumerate(zip(samples, channels)):
        lo, hi = max(int(s) - h, 0), min(int(s) + h + 1, filtered.shape[0])
        if hi - lo < 3:
            continue
        out[k] = (lo + int(np.argmin(filtered[lo:hi, c])) - int(s)) / fs * 1000.0
    return out
