"""Phase 9 regression tests: conditioners, noise-aware overlap rule, nuisance common-mode fit, perturbations, determinism."""

from __future__ import annotations

import unittest

import numpy as np

from edge.localize.overlap import spatial_overlap_conflict
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, P9_PARAMS, p9_factory
from edge.phase9.improved import (ConditionedLocalizer, RobustScanV2, condition, dead_channels, interpolate_dead,
                                  noise_aware_overlap)
from edge.phase9.synth_real import make_events, reference_array, xy_error
from edge.types import ElectrodeArray, ElectrodeSite, Snippet


def line_array(n=16, pitch=20.0):
    return ElectrodeArray(tuple(ElectrodeSite(i, (0.0, i * pitch, 0.0)) for i in range(n)))


def snippet(v, arr, n_pre=30):
    return Snippet(0, n_pre, 0, 30000.0, tuple(range(v.shape[1])), v, arr.xyz_um(), None, None)


class Conditioners(unittest.TestCase):
    def setUp(self):
        self.arr = line_array()
        rng = np.random.default_rng(1)
        v = rng.normal(0, 1.0, (75, 16)) + rng.normal(0, 30.0, (1, 16))      # static offsets
        v[30, 5] -= 60.0
        self.v = v

    def test_baseline_removes_static_offsets(self):
        out = condition(snippet(self.v, self.arr), baseline=True).voltages_uv
        self.assertLess(float(np.abs(np.median(out[:15], axis=0)).max()), 0.6)
        self.assertLess(out[30, 5], -50.0)                                   # spike survives

    def test_car_removes_shared_component(self):
        rng = np.random.default_rng(2)
        shared = rng.normal(0, 20.0, (75, 1))
        out = condition(snippet(self.v + shared, self.arr), baseline=True, car=True).voltages_uv
        self.assertLess(float(np.std(np.median(out, axis=1))), 1e-9)

    def test_dead_channels_detected_and_interpolated(self):
        v = self.v.copy(); v[:, 7] = 0.0
        sn = snippet(v, self.arr)
        self.assertTrue(dead_channels(v, sn.n_pre)[7])
        out = interpolate_dead(v, self.arr.xyz_um()[:, 1], dead_channels(v, sn.n_pre))
        self.assertTrue(np.allclose(out[:, 7], 0.5 * (v[:, 6] + v[:, 8])))
        self.assertTrue(np.array_equal(out[:, :7], v[:, :7]))

    def test_conditioning_is_pure_and_deterministic(self):
        sn = snippet(self.v, self.arr)
        before = sn.voltages_uv.copy()
        a = condition(sn, baseline=True, car=True, interp_dead=True).voltages_uv
        b = condition(sn, baseline=True, car=True, interp_dead=True).voltages_uv
        self.assertTrue(np.array_equal(a, b)); self.assertTrue(np.array_equal(sn.voltages_uv, before))


class OverlapRule(unittest.TestCase):
    def test_noise_peaks_do_not_reject_but_real_pair_does(self):
        arr = line_array(32)
        rng = np.random.default_rng(3)
        noise_only = rng.normal(0, 5.0, (75, 32)); noise_only[30, 3] -= 8.0; noise_only[30, 25] -= 8.0   # two weak bumps
        weak = snippet(noise_only, arr)
        self.assertFalse(noise_aware_overlap(weak, min_snr=6.0)[0])
        v = rng.normal(0, 2.0, (75, 32)); v[30, 3] -= 120.0; v[30, 25] -= 115.0
        pair = snippet(v, arr)
        self.assertTrue(noise_aware_overlap(pair, min_snr=6.0)[0])
        self.assertTrue(spatial_overlap_conflict(pair)[0])                   # the old rule agrees on a clear pair

    def test_robust_v2_reports_reason_when_rejecting(self):
        arr = line_array(32); rng = np.random.default_rng(4)
        v = rng.normal(0, 2.0, (75, 32)); v[30, 3] -= 120.0; v[30, 25] -= 115.0
        est = RobustScanV2(array=arr, min_snr=6.0).estimate(snippet(v, arr))
        self.assertFalse(est.available); self.assertIn("noise-aware", est.reason)


class NuisanceFitter(unittest.TestCase):
    def test_common_mode_does_not_create_second_source(self):
        arr = reference_array()
        _, ev = make_events("dev", n_per_noise=8, contaminated_fraction=0.0)
        plain = SpaceTimeFitter(arr, model="line_source", lambda_um=20.0, rank=4)
        nuis = SpaceTimeFitter(arr, model="line_source", lambda_um=20.0, rank=4, nuisance_cm=True)
        sn = [condition(e.snippet, baseline=True, interp_dead=True) for e in ev[:12]]
        dp = np.median([plain.fit(s).delta for s in sn]); dn = np.median([nuis.fit(s).delta for s in sn])
        self.assertLess(dn, 0.6 * dp)

    def test_nuisance_projection_is_orthogonal_to_common_mode(self):
        arr = line_array(16)
        f = SpaceTimeFitter(arr, nuisance_cm=True, model="line_source", lambda_um=20.0)
        rng = np.random.default_rng(5)
        sn = snippet(rng.normal(0, 3, (60, 16)), arr)
        Wt, sigma = f.whiten(sn)
        self.assertLess(float(np.abs(f._u @ Wt).max()), 1e-9)
        g = f._P(np.ones((3, 16)) / sigma)
        self.assertLess(float(np.abs(g @ f._u).max()), 1e-9)


class HarnessAndConfig(unittest.TestCase):
    def test_perturbations_are_seeded_and_change_signal(self):
        arr = line_array(); rng0 = np.random.default_rng(6)
        sn = snippet(rng0.normal(0, 3, (75, 16)), arr)
        for kind in ("independent", "correlated", "impulsive", "gain", "drift"):
            a = H.perturb(sn, kind, 1.0, np.random.default_rng(7)); b = H.perturb(sn, kind, 1.0, np.random.default_rng(7))
            self.assertTrue(np.array_equal(a.voltages_uv, b.voltages_uv)); self.assertFalse(np.array_equal(a.voltages_uv, sn.voltages_uv))
        with self.assertRaises(ValueError):
            H.perturb(sn, "bogus", 1.0, np.random.default_rng(0))

    def test_masks_keep_requested_fraction_and_clustered_is_contiguous(self):
        rng = np.random.default_rng(8)
        for frac in (0.75, 0.5):
            m = H.make_mask(32, frac, "random", rng); self.assertEqual(int(m.sum()), int(round(32 * frac)))
            c = H.make_mask(32, frac, "clustered", rng); drop = np.flatnonzero(~c)
            self.assertEqual(int(c.sum()), int(round(32 * frac))); self.assertTrue(np.all(np.diff(drop) == 1))

    def test_sub_snippet_matches_sub_array(self):
        arr = line_array(); keep = np.ones(16, bool); keep[[2, 3, 9]] = False
        sn = snippet(np.random.default_rng(9).normal(size=(75, 16)), arr)
        ss, sa = H.sub_snippet(sn, keep), H.sub_array(arr, keep)
        self.assertEqual(ss.voltages_uv.shape[1], sa.n_channels)
        self.assertTrue(np.allclose(ss.electrode_xyz_um, sa.xyz_um()))

    def test_p9_config_is_frozen_and_engines_conform(self):
        self.assertEqual(P9_PARAMS["lam_line_um"], 20.0)
        self.assertEqual(P9_HASH, "5ed8a69b97eb")
        arr = reference_array(); eng = p9_factory(arr)
        self.assertEqual(set(eng), {"amplitude", "template", "beamform", "weighted_scan", "robust_scan"})
        _, ev = make_events("dev", n_per_noise=2)
        for name, e in eng.items():
            est = e.estimate(ev[0].snippet)
            self.assertIn("available", est.to_dict()); self.assertIsInstance(e, ConditionedLocalizer)

    def test_real_like_synth_is_deterministic_and_pooled(self):
        _, a = make_events("dev", n_per_noise=3); _, b = make_events("dev", n_per_noise=3); _, c = make_events("select", n_per_noise=3)
        self.assertTrue(np.array_equal(a[0].snippet.voltages_uv, b[0].snippet.voltages_uv))
        self.assertNotEqual(a[0].truth_xy, c[0].truth_xy)
        self.assertGreater(xy_error((3.0, 4.0, 0.0), (0.0, 0.0)), 4.99)


if __name__ == "__main__":
    unittest.main()
