"""Phase 9: timestamp conversion and waveform-centering regression tests (no network)."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from edge.phase9.events import sorter_snippets, trough_offset_ms, window_array, window_sorter_seconds
from scripts.phase5.data import load_window, make_clock_maps, session_times_to_window_probe_seconds, window_path

FS = 30000.0


def synthetic_window(n_events=20, offset_s=-0.0015, sync_version=3, seed=3):
    rng = np.random.default_rng(seed)
    n_ch, n = 8, 30000
    v = rng.normal(0, 2.0, (n, n_ch))
    spikes = np.sort(rng.integers(3000, n - 3000, n_events))
    spikes = spikes[np.r_[True, np.diff(spikes) > 200]]
    for s in spikes:
        v[s - 1:s + 2, 3] += np.array([-20.0, -80.0, -30.0])          # trough exactly at sample s on channel 3
    sel = np.arange(8) * 2
    coords = np.stack([np.zeros(8) + 11, np.arange(8) * 20.0, np.zeros(8)], axis=1)
    return {"voltages_uv": v, "fs": FS, "sorter_times_s": spikes / FS, "sorter_clusters": np.zeros(len(spikes), int),
            "sorter_local_channel": np.full(len(spikes), 3), "selected": sel, "coordinates_um": coords,
            "pid": "x" * 8, "t0_s": 10.0, "duration_s": 1.0, "sync_offset_s": offset_s, "sync_version": sync_version}, spikes


class Alignment(unittest.TestCase):
    def test_probe_clock_snippets_are_centred_on_the_trough(self):
        w, spikes = synthetic_window()
        array, sn, meta = sorter_snippets(w, "probe")
        self.assertEqual(len(sn), len(spikes))
        for s in sn:
            col = s.voltages_uv[:, 3]
            self.assertEqual(int(np.argmin(col)), s.n_pre)

    def test_session_clock_error_is_visible_as_offset(self):
        w, spikes = synthetic_window(offset_s=-0.0015)
        off_probe = trough_offset_ms(w["voltages_uv"], spikes, np.full(len(spikes), 3), FS)
        shifted = np.round(window_sorter_seconds(w, "session") * FS).astype(int)
        off_session = trough_offset_ms(w["voltages_uv"], shifted, np.full(len(spikes), 3), FS)
        self.assertTrue(np.allclose(off_probe, 0.0))
        self.assertAlmostEqual(float(np.median(off_session)), 1.5, delta=0.05)     # 1.5 ms mis-centering

    def test_unsynced_window_is_refused(self):
        w, _ = synthetic_window(sync_version=2)
        with self.assertRaises(ValueError):
            window_sorter_seconds(w, "probe")

    def test_unknown_clock_rejected(self):
        w, _ = synthetic_window()
        with self.assertRaises(ValueError):
            window_sorter_seconds(w, "wall")

    def test_session_to_probe_round_trip_with_drift(self):
        samples = np.linspace(0, 3600 * FS, 100)
        table = np.stack([samples, samples / FS * (1 - 1e-5) + 0.002], axis=1)
        fwd, rev = make_clock_maps(table)
        for t in (100.0, 1465.0):
            sess = float(fwd(t * FS))
            rel = session_times_to_window_probe_seconds([sess], table, t - 0.5, FS)
            self.assertAlmostEqual(float(rel[0]), 0.5, places=6)

    def test_array_geometry_matches_selected_contacts(self):
        w, _ = synthetic_window()
        a = window_array(w)
        self.assertEqual(a.n_channels, 8)
        self.assertTrue(np.allclose(a.xyz_um()[:, 1], np.arange(8) * 20.0))


class CachedRealWindow(unittest.TestCase):
    def test_real_window_alignment_if_cached(self):
        p = window_path("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 100.0, 1.0)
        if not p.exists():
            self.skipTest("missing dataset: " + p.name)
        from scripts.real_detector import frozen_real_detector, preprocess
        w = load_window(p)
        filt, _ = preprocess(w["voltages_uv"], frozen_real_detector(w["fs"]))

        def med_abs(clock):
            s = np.round(window_sorter_seconds(w, clock) * w["fs"]).astype(int)
            k = (s > 90) & (s < filt.shape[0] - 90)
            o = trough_offset_ms(filt, s[k], np.asarray(w["sorter_local_channel"])[k], w["fs"])
            return float(np.nanmedian(np.abs(o)))
        self.assertLess(med_abs("probe"), 0.5)
        self.assertGreater(med_abs("session"), med_abs("probe") + 0.5)   # the historical clock is measurably worse


if __name__ == "__main__":
    unittest.main()
