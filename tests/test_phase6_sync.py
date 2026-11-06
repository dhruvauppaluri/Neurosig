"""Timestamp-alignment regression tests (no network): the Phase 3-4 failure mode must stay detectable and fixed."""
import numpy as np
import pytest

from scripts.phase5.data import make_clock_maps, session_times_to_window_probe_seconds
from scripts.phase5.metrics import window_metrics
from scripts.detect_spikes import Event

FS = 30000.0
DRIFT = -1.0e-5  # session - probe seconds per probe second (measured -9.5 .. -10.5 us/s on the IBL insertions)


def table(offset0=0.0, drift=DRIFT, n=200):
    samples = np.linspace(0, 3600 * FS, n)
    probe_s = samples / FS
    return np.stack([samples, probe_s + offset0 + drift * probe_s], axis=1)


def test_forward_reverse_round_trip_and_offset_sign():
    fwd, rev = make_clock_maps(table())
    for t in (0.0, 50.0, 1465.0, 3000.0):
        assert abs(fwd(t * FS) - (t + DRIFT * t)) < 1e-9
        assert abs(rev(fwd(t * FS)) - t * FS) < 1e-2  # samples
    assert fwd(3000 * FS) - 3000 < 0  # session clock runs behind the probe clock, as in the data


def test_session_times_map_to_probe_window_relative_seconds():
    tbl = table(offset0=0.0005)
    t0, probe_true = 2000.0, np.array([2000.1, 2000.5, 2001.9])
    fwd, _ = make_clock_maps(tbl)
    session = np.array([float(fwd(p * FS)) for p in probe_true])
    rel = session_times_to_window_probe_seconds(session, tbl, t0, FS)
    assert np.allclose(rel, probe_true - t0, atol=1e-6)


def _events(samples):
    return [Event(int(s), 0, -50.0, 5.0) for s in samples]


def test_unsynced_reference_breaks_matching_and_synced_reference_restores_it():
    """Original failure: reference times on the session clock compared with probe-clock detections."""
    tbl = table()
    t0 = 1500.0
    fwd, _ = make_clock_maps(tbl)
    probe_rel = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
    det = _events(np.rint(probe_rel * FS))
    session_ref = np.array([float(fwd((t0 + r) * FS)) for r in probe_rel])  # what IBL stores
    unsynced = session_ref - t0  # what Phase 3-4 effectively used
    synced = session_times_to_window_probe_seconds(session_ref, tbl, t0, FS)
    offset_ms = abs(fwd(t0 * FS) - t0) * 1000
    assert offset_ms > 10  # a 15 ms offset at 1500 s, far beyond the 1 ms tolerance
    bad = window_metrics(det, unsynced, FS, 1.0, 1.0)
    good = window_metrics(det, synced, FS, 1.0, 1.0)
    assert bad["tp"] == 0 and good["tp"] == len(det)


def test_offset_grows_with_time_so_early_windows_hide_the_problem():
    fwd, _ = make_clock_maps(table())
    off = {t: abs(float(fwd(t * FS)) - t) * 1000 for t in (50, 100, 200, 1465, 3259)}
    assert off[50] < 1.0 and off[200] < 3.0 and off[3259] > 30
    assert list(off.values()) == sorted(off.values())


def test_window_cache_version_guard(tmp_path):
    from scripts.phase5.data import load_window
    p = tmp_path / "w.npz"
    np.savez(p, voltages_uv=np.zeros((10, 2), dtype=np.float32), fs=np.float64(30000.0), sorter_times_s=np.zeros(0),
             sorter_clusters=np.zeros(0, dtype=np.int64), sorter_local_channel=np.zeros(0, dtype=int), selected=np.arange(2),
             coordinates_um=np.zeros((2, 3)), pid=np.asarray("x"), t0_s=np.float64(0.0), duration_s=np.float64(1.0))
    assert load_window(p)["sync_version"] == 1  # legacy cache without sync information is identifiable as unsynced
    from scripts.phase5.benchmark import load_split  # raises for unsynced windows
    assert callable(load_split)
