"""Phase 8 regression tests: geometry, assignment, overlap, uncertainty, rejection, dropout, determinism, schema."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from edge.benchmark_v3 import _array
from edge.forward import PointSource, render_recording
from edge.phase8.evaluate import folded_error_um
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase8.overlap_bench import assign, make_overlap_events, summarize_overlap
from edge.phase8.pipeline import Calibration, LocalizationResult, Phase8Localizer
from edge.phase8.registry import bootstrap_median_diff, error_stats, log_experiment
from edge.phase8.synth import make_events
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, ElectrodeSite, LocalizationEstimate

CAL = Calibration(tau_two=90.0, tau_gray=74.0, tau_shared=9.7, q_clean=8.0, q_flagged=30.0, q_two=29.0,
                  q_shared=18.0, max_unit_area_um2=1300.0, max_chi2_red=6.0)


def snippet_for(sources, array, noise, seed, truth=None):
    trace = render_recording(sources, array, 32000.0, noise, seed, 160)
    return extract_snippet(trace, 64, array, event_id=1, ground_truth_xyz_um=truth)


class GeometryAndCoordinates(unittest.TestCase):
    def test_folded_error_collinear_ignores_z(self):
        e = _array("linear").xyz_um()
        # (30, y, 40) and (50, y, 0) share rho=50 on a collinear probe
        self.assertAlmostEqual(folded_error_um((50.0, 100.0, 0.0), (30.0, 100.0, 40.0), e), 0.0, places=9)

    def test_folded_error_planar_uses_3d(self):
        e = _array("staggered").xyz_um()
        self.assertAlmostEqual(folded_error_um((0, 0, 0), (3, 4, 0), e), 5.0, places=9)

    def test_boundary_coordinates_stay_in_bounds(self):
        array = _array("linear")
        fitter = SpaceTimeFitter(array)
        for y in (0.0, 619.0):
            sn = snippet_for([PointSource(64, (20.0, y, 0.0), 12.0)], array, 2.0, 5, (20.0, y, 0.0))
            fit = fitter.fit(sn)
            s = fit.sources1[0]
            self.assertTrue(fitter.lo[0] <= s.x <= fitter.hi[0] and fitter.lo[1] <= s.y <= fitter.hi[1])
            self.assertLess(abs(s.y - y), 25.0)

    def test_missing_contact_array_is_handled(self):
        full = _array("linear")
        sites = tuple(ElectrodeSite(i, s.xyz_um) for i, s in enumerate(full.sites) if i % 4 != 1)
        array = ElectrodeArray(sites, "missing contacts")
        truth = (30.0, 300.0, 0.0)
        sn = snippet_for([PointSource(64, truth, 12.0)], array, 2.0, 7, truth)
        fit = SpaceTimeFitter(array).fit(sn)
        self.assertLess(abs(fit.sources1[0].y - 300.0), 15.0)


class SingleSourceAndDeterminism(unittest.TestCase):
    def setUp(self):
        self.array = _array("linear")
        self.truth = (35.0, 280.0, 0.0)
        self.sn = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 2.0, 11, self.truth)

    def test_single_source_accurate_and_single(self):
        r = Phase8Localizer(self.array, CAL).localize(self.sn)
        self.assertEqual(r.n_sources, 1)
        self.assertTrue(r.accepted)
        x, y, _ = r.primary.xyz_um
        self.assertLess(np.hypot(x - 35.0, y - 280.0), 6.0)

    def test_deterministic(self):
        loc = Phase8Localizer(self.array, CAL)
        a, b = loc.localize(self.sn).to_dict(), loc.localize(self.sn).to_dict()
        a.pop("latency_s"); b.pop("latency_s")
        self.assertEqual(a, b)

    def test_noise_perturbation_is_seeded(self):
        a = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 10.0, 99)
        b = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 10.0, 99)
        c = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 10.0, 100)
        self.assertTrue(np.array_equal(a.voltages_uv, b.voltages_uv))
        self.assertFalse(np.array_equal(a.voltages_uv, c.voltages_uv))


class UncertaintyAndRejection(unittest.TestCase):
    def setUp(self):
        self.array = _array("linear")
        self.truth = (35.0, 280.0, 0.0)

    def test_covariance_psd_and_shrinks_with_less_noise(self):
        fitter = SpaceTimeFitter(self.array, rank=4)
        areas = []
        for noise, seed in ((2.0, 3), (20.0, 3)):
            sn = snippet_for([PointSource(64, self.truth, 12.0)], self.array, noise, seed)
            Wt, sigma = fitter.whiten(sn)
            fit = fitter.fit(sn)
            s = fit.sources1[0]
            cov = fitter.covariance(np.array([[s.x, s.y]]), Wt, sigma)
            w = np.linalg.eigvalsh(0.5 * (cov + cov.T))
            self.assertTrue((w > -1e-9).all())
            areas.append(np.sqrt(max(np.linalg.det(cov), 0.0)))
        self.assertLess(areas[0], areas[1])

    def test_area_cap_rejects_with_reason(self):
        sn = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 25.0, 4)
        tight = Calibration(**{**CAL.to_json(), "max_unit_area_um2": 1e-3})
        r = Phase8Localizer(self.array, tight).localize(sn)
        self.assertFalse(r.accepted)
        self.assertIn("uncertainty area", r.reject_reason)
        est = Phase8Localizer(self.array, tight).estimate(sn)
        self.assertFalse(est.available)

    def test_zero_signal_rejected(self):
        sn = snippet_for([], self.array, 0.0, 1)
        r = Phase8Localizer(self.array, CAL).localize(sn)
        self.assertFalse(r.accepted)
        self.assertIn("flags", r.to_dict())

    def test_channel_dropout_mask_removes_influence(self):
        sn = snippet_for([PointSource(64, self.truth, 12.0)], self.array, 2.0, 21)
        v = np.array(sn.voltages_uv)
        mask = np.ones(32, bool); mask[:8] = False
        v2 = v.copy(); v2[:, :8] = 1e4 * np.random.default_rng(0).normal(size=(v.shape[0], 8))
        sn2 = type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v2,
                       sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)
        f = SpaceTimeFitter(self.array, rank=4)
        a = f.fit(sn, mask=mask).sources1[0]
        b = f.fit(sn2, mask=mask).sources1[0]
        self.assertAlmostEqual(a.y, b.y, delta=0.5)
        self.assertAlmostEqual(a.x, b.x, delta=0.5)


class OverlapAndAssignment(unittest.TestCase):
    def test_assign_penalizes_missed_extra_duplicate(self):
        truth = np.array([[10.0, 100.0], [10.0, 200.0]])
        self.assertEqual(assign(truth, truth)["matched"], 2)
        one = assign(truth, np.array([[10.0, 100.0]]))
        self.assertEqual((one["matched"], one["missed"], one["extra"]), (1, 1, 0))
        self.assertGreater(one["event_err"], 40.0)          # missed truth charged its distance (100 um)
        dup = assign(truth, np.array([[10.0, 100.0], [10.0, 101.0]]))
        self.assertEqual(dup["matched"], 1)                 # the duplicate cannot match the 2nd truth
        extra = assign(truth, np.array([[10.0, 100.0], [10.0, 200.0], [60.0, 150.0]]))
        self.assertEqual(extra["extra"], 1)
        none = assign(truth, np.zeros((0, 2)))
        self.assertEqual(none["missed"], 2)

    def test_pair_detected_and_single_not(self):
        array = _array("linear")
        p1, p2 = (30.0, 200.0, 0.0), (45.0, 330.0, 0.0)
        loc = Phase8Localizer(array, CAL)
        pair = loc.localize(snippet_for([PointSource(64, p1, 14.0), PointSource(70, p2, 12.0)], array, 2.0, 31))
        self.assertEqual(pair.n_sources, 2)
        ys = sorted(s.xyz_um[1] for s in pair.sources)
        self.assertAlmostEqual(ys[0], 200.0, delta=15.0)
        self.assertAlmostEqual(ys[1], 330.0, delta=15.0)
        single = loc.localize(snippet_for([PointSource(64, p1, 14.0)], array, 2.0, 32))
        self.assertEqual(single.n_sources, 1)

    def test_resolvable_definition_is_fixed(self):
        ev = make_overlap_events("dev", n_per_noise=3)
        for e in ev:
            if e.resolvable:
                self.assertEqual(e.kind, "pair")
                self.assertGreaterEqual(e.sep_um, 60.0)
                self.assertGreaterEqual(min(e.snr), 6.0)
                self.assertGreaterEqual(e.ratio, 1 / 3 - 1e-9)

    def test_summary_runs(self):
        ev = make_overlap_events("dev", n_per_noise=2)
        recs = [{"n_est": 1, "rejected": False, **assign(np.asarray(e.truths_rho_y), np.asarray([e.truths_rho_y[0]]))}
                for e in ev]
        s = summarize_overlap(ev, recs)
        self.assertEqual(s["n"]["events"], len(ev))


class SchemaAndRegistry(unittest.TestCase):
    def test_result_dict_is_superset_of_legacy_keys(self):
        array = _array("linear")
        sn = snippet_for([PointSource(64, (30.0, 250.0, 0.0), 12.0)], array, 2.0, 41)
        loc = Phase8Localizer(array, CAL)
        d = loc.localize(sn).to_dict()
        legacy = {"available", "method", "xyz_um", "confidence", "reason", "event_id", "unverified"}
        self.assertTrue(legacy <= set(d))
        est = loc.estimate(sn)
        self.assertIsInstance(est, LocalizationEstimate)
        self.assertEqual(set(est.to_dict()), {"available", "method", "xyz_um", "ground_truth_xyz_um", "error_um",
                                              "latency_s", "confidence", "reason", "event_id", "unverified"})

    def test_pools_are_disjoint(self):
        a = make_events("dev", n_per_cell=2)
        b = make_events("select", n_per_cell=2)
        self.assertFalse(np.allclose(a[0].snippet.voltages_uv, b[0].snippet.voltages_uv))
        self.assertNotEqual(a[0].truth_xyz_um, b[0].truth_xyz_um)

    def test_registry_requires_fields_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            with self.assertRaises(ValueError):
                log_experiment({"id": "x"}, p)
            log_experiment({"id": "x", "hypothesis": "h", "split": "s", "params": {"a": 1}, "before": {}, "after": {},
                            "decision": "retain", "reasoning": "r"}, p)
            self.assertIn("config_hash", p.read_text())
        s = error_stats([1.0, 2.0, 3.0, None, float("nan")])
        self.assertEqual(s["n"], 3)
        b = bootstrap_median_diff([5, 6, 7, 8], [1, 2, 3, 4])
        self.assertGreater(b["median_diff"], 0)


if __name__ == "__main__":
    unittest.main()
