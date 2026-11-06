"""Phase 5 window cache for real IBL AP recordings.

Windows are streamed once through ``edge.ibl.load_raw_chunk`` (needs the IBL
venv) and cached as compressed ``.npz`` under ``data/phase5/windows`` (Git
ignored). Everything downstream only needs numpy.

Sorter spikes are an imperfect reference; they are stored, not interpreted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "phase5" / "windows"
MANIFEST = ROOT / "reports" / "phase5" / "split_manifest.json"


def window_path(pid: str, t0_s: float, duration_s: float) -> Path:
    return CACHE / f"{pid[:8]}_{t0_s:09.3f}_{duration_s:.1f}.npz"


SYNC_VERSION = 3  # sorter times converted session clock -> probe sample clock


def _one():
    from one.api import ONE
    return ONE(base_url="https://openalyx.internationalbrainlab.org",
               username="intbrainlab", password="international", silent=True,
               cache_dir="data/neuropixels/ibl_cache")


def sorter_in_probe_time(one, pid: str, t0_s: float, duration_s: float, selected: np.ndarray, fs: float):
    """Sorter spikes of the selected contacts, in *probe* seconds relative to ``t0_s``.

    IBL ``spikes.times`` are on the session main clock. Raw AP samples are on the
    probe clock; ``_spikeglx_*.timestamps.npy`` maps probe sample -> session
    seconds. Comparing session-clock spike times with raw sample indices (as the
    earlier phases did) is only valid near the recording start.
    """
    from scipy.interpolate import interp1d
    from edge.ibl import SORTING_REVISION

    eid, probe = one.pid2eid(pid)
    col = f"alf/{probe}/pykilosort"
    ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy",
                                     collection=f"raw_ephys_data/{probe}"), dtype=float)
    forward = interp1d(ts[:, 0], ts[:, 1], fill_value="extrapolate")
    reverse = interp1d(ts[:, 1], ts[:, 0], fill_value="extrapolate")
    a, b = t0_s * fs, (t0_s + duration_s) * fs
    sess_lo, sess_hi = float(forward(a)), float(forward(b))
    st = np.asarray(one.load_dataset(eid, "spikes.times.npy", collection=col, revision=SORTING_REVISION), dtype=float)
    sc = np.asarray(one.load_dataset(eid, "spikes.clusters.npy", collection=col, revision=SORTING_REVISION), dtype=int)
    cc = np.asarray(one.load_dataset(eid, "clusters.channels.npy", collection=col, revision=SORTING_REVISION), dtype=int)
    m = (st >= sess_lo) & (st < sess_hi)
    st, sc = st[m], sc[m]
    sel = list(int(c) for c in selected)
    peak = cc[sc]
    keep = np.isin(peak, sel)
    st, sc, peak = st[keep], sc[keep], peak[keep]
    probe_t = reverse(st) / fs - t0_s
    local = np.asarray([sel.index(int(c)) for c in peak], dtype=int)
    return probe_t, sc, local, float(forward(a) - t0_s)


def resync_window(path: Path) -> Path:
    """Replace the sorter arrays of a cached window with probe-clock times."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if int(data.get("sync_version", 1)) >= SYNC_VERSION:
        return path
    t, c, loc, off = sorter_in_probe_time(_one(), str(data["pid"]), float(data["t0_s"]),
                                          float(data["duration_s"]), data["selected"], float(data["fs"]))
    data.update(sorter_times_s=t.astype(np.float64), sorter_clusters=c.astype(np.int64),
                sorter_local_channel=loc, sync_offset_s=np.float64(off), sync_version=np.int64(SYNC_VERSION))
    np.savez_compressed(path, **data)
    return path


def fetch_window(pid: str, t0_s: float, duration_s: float, n_channels: int = 32) -> Path:
    """Stream and cache one window. Idempotent."""
    path = window_path(pid, t0_s, duration_s)
    if path.exists():
        return resync_window(path)
    from edge.ibl import load_raw_chunk  # needs IBL environment

    chunk = load_raw_chunk(pid, t0_s, duration_s, n_channels)
    times, clusters, local, offset = sorter_in_probe_time(
        chunk["one"], pid, t0_s, duration_s, chunk["selected"], float(chunk["fs"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        voltages_uv=chunk["voltages_uv"].astype(np.float32),
        fs=np.float64(chunk["fs"]),
        sorter_times_s=times.astype(np.float64),
        sorter_clusters=clusters.astype(np.int64),
        sorter_local_channel=local,
        selected=np.asarray(chunk["selected"], dtype=int),
        coordinates_um=chunk["array"].xyz_um(),
        pid=np.asarray(pid),
        t0_s=np.float64(t0_s),
        duration_s=np.float64(duration_s),
        sync_offset_s=np.float64(offset),
        sync_version=np.int64(SYNC_VERSION),
    )
    return path


def load_window(path: Path | str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {
            "voltages_uv": z["voltages_uv"].astype(np.float64),
            "fs": float(z["fs"]),
            "sorter_times_s": z["sorter_times_s"],
            "sorter_clusters": z["sorter_clusters"],
            "sorter_local_channel": z["sorter_local_channel"],
            "selected": z["selected"],
            "coordinates_um": z["coordinates_um"],
            "pid": str(z["pid"]),
            "t0_s": float(z["t0_s"]),
            "duration_s": float(z["duration_s"]),
            "sync_offset_s": float(z["sync_offset_s"]) if "sync_offset_s" in z.files else None,
            "sync_version": int(z["sync_version"]) if "sync_version" in z.files else 1,
        }


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def manifest_windows(split: str) -> list[dict]:
    """Windows of one split: 'calibration', 'selection', 'heldout', or 'seen'."""
    return [w for w in load_manifest()["windows"] if w["split"] == split]
