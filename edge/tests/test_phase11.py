"""Phase 11 regression tests: event referencing and matching, multichannel alignment, null generation, SNR,
forward models, missing contacts, count rule, calibration, P11 config equivalence, final-manifest separation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from edge.phase11 import calibration as CAL
from edge.phase11 import forward_models as F
from edge.types import ElectrodeArray, ElectrodeSite, Snippet

ROOT = Path(__file__).resolve().parents[2]
HAVE_WINDOWS = any((ROOT / "data/phase5/windows").glob("*.npz")) if (ROOT / "data/phase5/windows").exists() else False


def grid_E(n=32):
    return np.array([[x, 20.0 * (i // 2), 0.0] for i, x in zip(range(n), [11.0, 43.0] * n)], float)


class ForwardModels(unittest.TestCase):
    def test_line20_recovers_source_and_is_deterministic(self):
        E = grid_E(); a = 120 * F.line_gain(30.0, 150.0, E, 20.0); s = np.full(32, 2.0)
        f1 = F.fit(F.Model("line20"), a, s, E); f2 = F.fit(F.Model("line20"), a, s, E)
        self.assertLess(np.hypot(f1.theta[0] - 30, f1.theta[1] - 150), 1.0)
        self.assertTrue(np.allclose(f1.theta, f2.theta))

    def test_mask_excludes_contacts(self):
        E = grid_E(); a = 120 * F.line_gain(30.0, 150.0, E, 20.0); s = np.full(32, 2.0)
        bad = a.copy(); bad[:6] = 1e4
        m = np.ones(32, bool); m[:6] = False
        f = F.fit(F.Model("line20"), bad, s, E, mask=m)
        self.assertLess(np.hypot(f.theta[0] - 30, f.theta[1] - 150), 2.0)

    def test_linefree_distance_bounded_and_offset_model_signed(self):
        E = grid_E(); a = 100 * F.line_gain(20.0, 200.0, E, 45.0) - 5.0; s = np.full(32, 1.0)
        f = F.fit(F.Model("linefree"), a + 5.0, s, E)
        self.assertTrue(5.0 <= f.theta[2] <= 80.0)
        self.assertAlmostEqual(f.theta[2], 45.0, delta=6.0)
        b = 100 * F.line_gain(20.0, 200.0, E, 20.0) - 5.0
        g = F.fit(F.Model("line20c", n_src=2), b, s, E)
        self.assertAlmostEqual(g.amps[1], -5.0, delta=0.5)    # recovers the negative offset

    def test_learned_radial_is_monotone_normalized(self):
        r = np.linspace(0, 300, 400); v = 20 / np.sqrt(r ** 2 + 400)
        t = F.learn_radial([(r, v)])
        self.assertAlmostEqual(t[0], 1.0)
        self.assertTrue(np.all(np.diff(t) <= 1e-12))


class Calibration(unittest.TestCase):
    def test_conformal_quantile_and_pooling(self):
        rng = np.random.default_rng(0)
        rows = [{"d2": float(x), "stratum": "clean", "snr_g": "high", "drop": "none"} for x in rng.chisquare(2, 400)]
        rows += [{"d2": float(x), "stratum": "clean", "snr_g": "low", "drop": "dead"} for x in 4 * rng.chisquare(2, 10)]
        m = CAL.fit_groups(rows)
        self.assertEqual(m["info"][("clean", "high", "none")][0], "own")
        self.assertEqual(m["info"][("clean", "low", "dead")][0], "pooled_conservative")
        self.assertGreaterEqual(m["table"][("clean", "low", "dead")], m["stratum"]["clean"])
        self.assertAlmostEqual(m["table"][("clean", "high", "none")], float(np.quantile([r["d2"] for r in rows[:400]], 0.9)), delta=0.6)

    def test_groups_and_wilson(self):
        self.assertEqual(CAL.snr_group(5), "low"); self.assertEqual(CAL.snr_group(10), "mid"); self.assertEqual(CAL.snr_group(30), "high")
        self.assertEqual(CAL.stratum(["shared_waveform_pair"], 2), "shared"); self.assertEqual(CAL.stratum([], 2), "two")
        lo, hi = CAL.wilson(90, 100); self.assertTrue(lo < 0.9 < hi)


@unittest.skipUnless(HAVE_WINDOWS, "needs cached development windows")
class Referencing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from edge.phase9 import harness as H
        from edge.phase11.referencing import load_events
        from scripts.phase5.data import load_window, window_path
        w = H.manifest_windows("dev")[0]
        cls.we = load_events(w, load_window(window_path(w["pid"], w["t0_s"], w["duration_s"])))

    def test_probe_clock_times_and_detector_matching(self):
        from edge.phase11.referencing import chance_matches, match_ab
        we = self.we
        pairs = match_ab(we, we.det_sample)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(all(abs(dt) <= 15 for _, _, dt in pairs))                     # 0.5 ms at 30 kHz
        self.assertEqual(len({d for d, _, _ in pairs}), len(pairs))                  # one-to-one
        self.assertGreater(len(pairs), 2 * chance_matches(we, we.det_sample))
        # detector events shifted by 20 ms do not match beyond chance
        far = match_ab(we, we.det_sample + 600)
        self.assertLess(len(far), 0.5 * len(pairs))

    def test_alignment_is_bounded_and_common_to_channels(self):
        from edge.phase11.referencing import ALIGN_HALF, align_sample, align_sample_raw
        we = self.we
        for s, c in list(zip(we.sorter_sample, we.sorter_channel))[:50]:
            a = align_sample(we, int(s), int(c)); b = align_sample_raw(we, int(s), int(c), 3)
            self.assertLessEqual(abs(a - s), ALIGN_HALF); self.assertLessEqual(abs(b - s), 3)

    def test_snr_and_null(self):
        from edge.phase11.referencing import event_quality
        we = self.we
        real = np.median([event_quality(we, int(s), int(c))["snr"] for s, c in zip(we.sorter_sample[:80], we.sorter_channel[:80])])
        null = np.median([event_quality(we, int(s) + 600, int(c))["snr"] for s, c in zip(we.sorter_sample[:80], we.sorter_channel[:80]) if s + 700 < we.raw.shape[0]])
        self.assertGreater(real, null)


@unittest.skipUnless(HAVE_WINDOWS, "needs a cached development window for the real geometry")
class ConfigsAndOverlap(unittest.TestCase):
    def test_p11_identical_on_intact_and_rejects_ambiguous_dead(self):
        from edge.phase10.configs import p10_factory
        from edge.phase11.configs import P11_HASH, p11_factory
        from edge.phase9.synth_real import make_events
        array, ev = make_events("dev", n_per_noise=1)
        f10, f11 = p10_factory(array), p11_factory(array)
        for e in ev:
            for k in f10:
                self.assertEqual(f10[k].estimate(e.snippet).xyz_um, f11[k].estimate(e.snippet).xyz_um)
        sn = ev[0].snippet; v = np.asarray(sn.voltages_uv, float).copy(); v[:, :20] = 0.0
        z = Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um, sn.peak_channel, None)
        r = f11["weighted_scan"].estimate(z)
        self.assertTrue((not r.available and "dead-contact" in r.reason) or r.available)
        self.assertLessEqual(len(f11["weighted_scan"]._cache), 4)
        self.assertEqual(len(P11_HASH), 12)

    def test_axial_only_boundary_rule_keeps_lateral_pairs(self):
        from edge.phase10.overlap_stress import localizer
        from edge.phase11.overlap import CountRule, localize
        from edge.phase9.improved import condition
        from edge.phase9.synth_real import FS, _render, reference_array
        from edge.snippets import extract_snippet
        array = reference_array(); E = array.xyz_um(); rng = np.random.default_rng(2)
        y0 = float(np.median(E[:, 1]))
        src = [(10.0, y0 - 50, 5.0, 150.0, 60), (85.0, y0 + 50, 5.0, 140.0, 66)]
        sn = extract_snippet(_render(src, E, 3.0, rng, 20.0, 15.0, 0.5), 60, array, event_id=0, sample_rate_hz=FS)
        loc = localizer(array)
        r, _ = localize(loc, condition(sn, baseline=True, interp_dead=True), CountRule(axial_only_boundary=True))
        self.assertNotIn("boundary_pair_rejected", r.flags)
        self.assertEqual(r.n_sources, 2)
        single = [(30.0, y0, 5.0, 150.0, 60)]
        sn1 = extract_snippet(_render(single, E, 3.0, rng, 20.0, 15.0, 0.5), 60, array, event_id=1, sample_rate_hz=FS)
        r1, _ = localize(loc, condition(sn1, baseline=True, interp_dead=True), CountRule(axial_only_boundary=True))
        self.assertEqual(r1.n_sources, 1)


class Manifests(unittest.TestCase):
    def test_final_sessions_new_and_distinct(self):
        m11 = json.loads((ROOT / "reports/phase11/split_manifest_phase11.json").read_text())
        used = set()
        for p in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json", "reports/phase10/split_manifest_phase10.json"):
            used |= {v["pid"] for v in json.loads((ROOT / p).read_text())["sessions"].values()}
        new = {v["pid"] for v in m11["sessions"].values()}
        self.assertEqual(len(new), 4); self.assertFalse(new & used)
        self.assertEqual(len(m11["windows"]), 24)


if __name__ == "__main__":
    unittest.main()
