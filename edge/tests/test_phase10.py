"""Phase 10 regression tests: segment/SNR metrics, overlap stress helpers, K=3 extension, dead-contact exclusion,
P10 equivalence with P9, and the fresh locked manifest."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from edge.phase10.overlap_stress import (StressEvent, _resolvable, detected_dead_mask, evaluate_event, k3_fit,
                                         localizer, match)
from edge.phase10.realextra import _tertiles, event_snr, summarize
from edge.phase9.synth_real import FS, _render, reference_array
from edge.snippets import extract_snippet
from edge.types import Snippet

ROOT = Path(__file__).resolve().parents[2]
HAVE_WINDOWS = (ROOT / "data/phase5/windows").exists() and any((ROOT / "data/phase5/windows").glob("*.npz"))


def _zeroed(sn, cols):
    v = np.asarray(sn.voltages_uv, float).copy(); v[:, cols] = 0.0
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um,
                   sn.peak_channel, sn.ground_truth_xyz_um)


class SegmentAndSnr(unittest.TestCase):
    def test_tertiles_split_by_window_time(self):
        rows = [{"sess": "X", "t0": t} for t in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)]
        lab = _tertiles(rows)
        self.assertEqual([lab[("X", t)] for t in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)], [0, 0, 1, 1, 2, 2])

    def test_segment_shift_detects_a_drifting_unit(self):
        rows = []
        for k, t in enumerate((1.0, 2.0, 3.0)):
            for j in range(4):
                rows.append({"sess": "X", "unit": ("X", "p", 1), "t0": t, "snr": 10.0 + j,
                             "xy": np.array([0.0, 100.0 + 30.0 * k + 0.1 * j])})
        s = summarize(rows)
        self.assertAlmostEqual(s["segment_shift_um"]["median"], 60.0, delta=0.5)
        self.assertEqual(s["n_units_ge5"], 1)

    def test_event_snr_positive_for_trough(self):
        v = np.random.default_rng(0).normal(0, 1.0, (75, 4)); v[25:28, 2] -= 20.0
        sn = Snippet(0, 25, 0, 30000.0, (0, 1, 2, 3), v, np.zeros((4, 3)), 2, None)
        snr, peak = event_snr(sn, 2)
        self.assertEqual(peak, 2)
        self.assertGreater(snr, 8.0)


class OverlapStressHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.array = reference_array() if HAVE_WINDOWS else None

    def test_match_counts_missed_and_extra(self):
        t = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
        m = match(t, np.array([[0.0, 1.0], [0.0, 400.0]]))
        self.assertEqual((m["matched"], m["extra"], m["missed"]), (1, 1, 2))
        self.assertEqual(match(t, np.zeros((0, 2)))["missed"], 3)

    def test_resolvable_definition(self):
        E = np.array([[0.0, y, 0.0] for y in range(0, 400, 20)])
        good = [(10, 100, 0, 120, 60), (10, 200, 0, 100, 63)]
        self.assertTrue(_resolvable(good, E, 5.0))
        self.assertFalse(_resolvable([(10, 100, 0, 120, 60), (10, 140, 0, 100, 63)], E, 5.0))   # < 60 um
        self.assertFalse(_resolvable([(10, 100, 0, 120, 60), (10, 200, 0, 30, 63)], E, 5.0))    # ratio < 1/3

    @unittest.skipUnless(HAVE_WINDOWS, "needs a cached dev window for the real geometry")
    def test_dead_mask_detects_zeroed_contacts_and_avoids_phantom_source(self):
        E = self.array.xyz_um(); rng = np.random.default_rng(3)
        y = float(np.median(E[:, 1]))
        src = [(30.0, y, 10.0, 120.0, 60)]
        tr = _render(src, E, 5.0, rng, 20.0, 15.0, 0.5)
        sn = extract_snippet(tr, 60, self.array, event_id=0, sample_rate_hz=FS)
        cols = np.arange(4, 12)
        z = _zeroed(sn, cols)
        m = detected_dead_mask(z)
        self.assertIsNotNone(m)
        self.assertTrue((~m[cols]).all() and m.sum() == len(m) - len(cols))
        ev = StressEvent("single", 5.0, tuple(src), 0, 0.0, False, z)
        r = evaluate_event(localizer(self.array), ev, dead_as_mask=True)
        self.assertEqual(r["n_est"], 1)
        self.assertEqual(r["matched"], 1)

    @unittest.skipUnless(HAVE_WINDOWS, "needs a cached dev window for the real geometry")
    def test_k3_fit_recovers_well_separated_triple(self):
        E = self.array.xyz_um(); rng = np.random.default_rng(4)
        y0 = float(E[:, 1].min()) + 60
        src = [(20.0, y0, 5.0, 150.0, 60), (25.0, y0 + 100, 5.0, 130.0, 65), (30.0, y0 + 200, 5.0, 120.0, 56)]
        tr = _render(src, E, 3.0, rng, 20.0, 15.0, 0.5)
        sn = extract_snippet(tr, 60, self.array, event_id=0, sample_rate_hz=FS)
        loc = localizer(self.array)
        from edge.phase9.improved import condition
        c = condition(sn, baseline=True, interp_dead=True)
        r = loc.localize(c)
        self.assertEqual(r.n_sources, 2)
        p3, d3 = k3_fit(loc, c, np.array([s.xyz_um[:2] for s in r.sources]))
        self.assertGreater(d3, 0.0)
        self.assertEqual(match(np.array([s[:2] for s in src]), p3)["matched"], 3)


@unittest.skipUnless(HAVE_WINDOWS, "needs a cached dev window for the real geometry")
class P10Config(unittest.TestCase):
    def test_identical_to_p9_without_dead_contacts_and_differs_with(self):
        from edge.phase9.configs import p9_factory
        from edge.phase9.synth_real import make_events
        from edge.phase10.configs import P10_HASH, p10_factory
        array, ev = make_events("dev", n_per_noise=1)
        f9, f10 = p9_factory(array), p10_factory(array)
        for e in ev:
            for k in f9:
                self.assertEqual(f9[k].estimate(e.snippet).xyz_um, f10[k].estimate(e.snippet).xyz_um)
        z = _zeroed(ev[0].snippet, np.arange(0, 8))
        r = f10["weighted_scan"].estimate(z)
        self.assertTrue(r.available and np.all(np.isfinite(r.xyz_um)))
        self.assertEqual(len(f10["weighted_scan"]._cache), 1)
        self.assertEqual(len(P10_HASH), 12)


class LockedManifest(unittest.TestCase):
    def test_phase10_locked_sessions_are_new(self):
        m10 = json.loads((ROOT / "reports/phase10/split_manifest_phase10.json").read_text())
        used = set()
        for p in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json"):
            used |= {v["pid"] for v in json.loads((ROOT / p).read_text())["sessions"].values()}
        new = {v["pid"] for v in m10["sessions"].values()}
        self.assertEqual(len(new), 4)
        self.assertFalse(new & used)
        self.assertEqual(len(m10["windows"]), 24)
        self.assertEqual(len({v["lab"] for v in m10["sessions"].values()}), 4)


if __name__ == "__main__":
    unittest.main()
