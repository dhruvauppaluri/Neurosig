"""Phase 12/13 regression tests: fast line20 equivalence, E3 alignment bounds, pooling estimators, count decision
logic, logistic count model, analytic-Jacobian K<=2 refine equivalence, dictionary lattice placement, sorter-free
grouping, final-manifest separation, frozen configuration, Phase 13 decoding utilities."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from edge.phase11 import forward_models as F
from edge.phase12 import fast as FA, pooling as PL
from edge.types import ElectrodeArray, ElectrodeSite

ROOT = Path(__file__).resolve().parents[2]


def checker(rows=16):
    """Neuropixels 1.0 checkerboard, 32 contacts."""
    E = []
    for r in range(rows):
        xs = (11.0, 43.0) if r % 2 == 0 else (27.0, 59.0)
        E += [[x, 20.0 * r + 100.0, 0.0] for x in xs]
    return np.array(E)


class FastLine20(unittest.TestCase):
    def test_matches_reference_fit(self):
        E = checker(); rng = np.random.default_rng(0)
        fl = FA.FastLine20(E)
        for _ in range(20):
            x, y = rng.uniform(5, 60), rng.uniform(140, 380)
            s = np.full(32, 3.0); a = 120 * F.line_gain(x, y, E, 20.0) + rng.normal(0, 3.0, 32)
            ref = F.fit(F.Model("line20"), a, s, E).theta[:2]
            th, area, chi = fl.fit(a, s, iters=5, area=True)
            self.assertLess(np.linalg.norm(th - ref), 2.0)
            self.assertGreater(area, 0)

    def test_e3_alignment_bounded(self):
        E = checker(); rng = np.random.default_rng(1)
        rc = rng.normal(0, 1, (200, 32)); rc[100, 5] -= 30
        s3 = FA.e3_align(rc, 98, 5, E, np.ones(32))
        self.assertTrue(95 <= s3 <= 101)
        self.assertEqual(s3, 100)


class Pooling(unittest.TestCase):
    def test_huber_mean_resists_outlier(self):
        X = np.ones((20, 5, 3)); X[0] *= 100
        self.assertLess(np.abs(PL.huber_mean(X) - 1).max(), np.abs(X.mean(0) - 1).max())

    def test_joint_line20_with_variable_amplitudes(self):
        E = checker(); rng = np.random.default_rng(2)
        A = np.array([amp * F.line_gain(30.0, 250.0, E, 20.0) + rng.normal(0, 2, 32) for amp in rng.uniform(40, 200, 15)])
        th = PL.joint_line20(A, np.full(32, 2.0), E)
        self.assertLess(np.hypot(th[0] - 30, th[1] - 250), 2.0)

    def test_geomedian_and_pairwise(self):
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [100.0, 100.0]])
        m = PL.weighted_geomedian(P, np.ones(4))
        self.assertLess(np.linalg.norm(m), 1.5)
        self.assertAlmostEqual(PL._pairwise([np.zeros(2), np.array([3.0, 4.0])]), 5.0)
        self.assertIsNone(PL._pairwise([None, np.zeros(2)]))


class CountRules(unittest.TestCase):
    def _a(self, pos2, delta=500.0, dshared=10.0):
        return {"pos1": np.array([[30.0, 200.0]]), "pos2": np.array(pos2), "pos_sh": np.array(pos2), "cov_pos1": np.eye(2),
                "cov_pos2": np.eye(4), "cov_pos_sh": np.eye(4), "en2": [2.0, 1.0], "en_sh": [2.0, 1.0], "delta": delta,
                "dshared": dshared, "lo": np.array([-30.0, 60.0]), "hi": np.array([100.0, 440.0]), "delta_norm": 50.0,
                "delta_shared_norm": 1.0, "x": np.zeros(10)}

    def test_merge_and_boundary(self):
        from edge.phase10.overlap_stress import frozen_calibration
        from edge.phase12 import count_model as CM
        cal = frozen_calibration()
        self.assertEqual(CM.finish(self._a([[30, 200], [35, 205]]), True, True, cal).flags, ["merged_close_pair"])
        self.assertEqual(CM.finish(self._a([[30, 200], [40, 440]]), True, True, cal).flags, ["boundary_pair_rejected"])
        d = CM.finish(self._a([[30, 200], [40, 300]]), True, True, cal)
        self.assertEqual(d.n, 2); self.assertEqual(d.cov.shape, (4, 4))

    def test_logistic_threshold_controls_singles(self):
        from edge.phase12 import count_model as CM
        rng = np.random.default_rng(3)
        Xs = rng.normal(0, 1, (400, 3)); Xp = rng.normal(2, 1, (200, 3))
        m = CM.LogisticCount().fit(np.vstack([Xs, Xp]), np.r_[np.zeros(400), np.ones(200)], Xs, target_fp=0.05)
        self.assertLessEqual(np.mean(m.prob(Xs) > m.thr), 0.06)
        m2 = CM.LogisticCount.from_json(m.to_json())
        self.assertTrue(np.allclose(m.prob(Xp), m2.prob(Xp)))


class FastSpaceTime(unittest.TestCase):
    def test_refine_matches_reference(self):
        from edge.phase10.overlap_stress import localizer
        from edge.phase9.improved import condition
        from edge.phase9.synth_real import FS, _render
        from edge.snippets import extract_snippet
        E = checker(); arr = ElectrodeArray(tuple(ElectrodeSite(i, tuple(E[i])) for i in range(32)), "t")
        rng = np.random.default_rng(4)
        ref, fast = localizer(arr), FA.fast_localizer(arr)
        for _ in range(5):
            src = [(float(rng.uniform(5, 60)), float(rng.uniform(150, 350)), 5.0, 140.0, 60)]
            sn = condition(extract_snippet(_render(src, E, 5.0, rng, 20.0, 15.0, 0.5), 60, arr, sample_rate_hz=FS), baseline=True)
            a = ref.fitter.fit(sn); b = fast.fitter.fit(sn)
            self.assertLess(np.hypot(a.sources1[0].x - b.sources1[0].x, a.sources1[0].y - b.sources1[0].y), 1.0)
            self.assertAlmostEqual(a.delta_shared, b.delta_shared, delta=max(1e-3 * abs(a.delta_shared), 0.5))


class Dictionary(unittest.TestCase):
    def test_lattice_placement_and_mirror(self):
        from edge.phase12 import dictionary as DI
        E = checker(); W = np.zeros((100, 32)); pk = 10
        W[30:60, pk] = -np.hanning(30) * 100
        a = DI.make_atom(W, E, 45, np.array([E[pk, 0], E[pk, 1]]), "t")
        dic = DI.Dictionary([a])
        G, P = dic.candidates(E)
        self.assertGreater(len(G), 8)                     # placements on both row types via the mirror
        self.assertTrue(np.all(np.isin(np.round(P[:, 0]), [11, 27, 43, 59])))


class OnlineGroups(unittest.TestCase):
    def test_identical_waveforms_group(self):
        from edge.phase12.online_groups import OnlineGrouper
        E = checker(); rng = np.random.default_rng(5)
        rc = rng.normal(0, 0.5, (3000, 32))
        for s in range(100, 2900, 100):
            rc[s - 3:s + 6, 8] += -40 * np.hanning(9); rc[s - 3:s + 6, 9] += -20 * np.hanning(9)
        g = OnlineGrouper(E)
        ids = [g.add(rc, s, 8, s) for s in range(100, 2900, 100)]
        self.assertEqual(len(set(ids)), 1)


class Manifests(unittest.TestCase):
    def test_final_sessions_new(self):
        m12 = json.loads((ROOT / "reports/phase12/split_manifest_phase12.json").read_text())
        used = set()
        for p in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json",
                  "reports/phase10/split_manifest_phase10.json", "reports/phase11/split_manifest_phase11.json"):
            used |= {v["pid"] for v in json.loads((ROOT / p).read_text())["sessions"].values()}
        new = {v["pid"] for v in m12["sessions"].values()}
        self.assertEqual(len(new), 3); self.assertFalse(new & used); self.assertEqual(len(m12["windows"]), 24)

    def test_frozen_hash(self):
        from edge.phase12.configs import P12_HASH
        self.assertEqual(len(P12_HASH), 12)


class Phase13Decode(unittest.TestCase):
    def test_folds_and_maps(self):
        from edge.phase13 import decode as D
        f = D.blocked_folds(103)
        self.assertEqual(len(np.unique(f)), 5); self.assertTrue(np.all(np.diff(f) >= 0))
        C = np.ones((4, 3, 2)); dep = np.array([10.0, 50.0, 90.0]); edges = np.array([0.0, 40.0, 80.0, 120.0])
        M = D.unit_depth_maps(C, dep, edges)
        self.assertTrue(np.allclose(M.sum(1), C.sum((1, 2))))
        per = [(np.array([0, 1]), np.array([0, 1]), np.array([20.0, 60.0]))] * 4
        a = D.spike_depth_maps(per, 4, edges, 2, True); self.assertEqual(a.shape, (4, 6))
        r1 = D.spike_depth_maps(per, 4, edges, 2, False, 30.0, np.random.default_rng(0))
        r2 = D.spike_depth_maps(per, 4, edges, 2, False, 30.0, np.random.default_rng(0))
        self.assertTrue(np.array_equal(r1, r2))

    def test_paired_boot_sign(self):
        from edge.phase13 import decode as D
        y = np.zeros(200, int); a = np.ones(200, int); b = np.zeros(200, int)
        r = D.paired_boot(y, a, b)
        self.assertAlmostEqual(r["delta"], 1.0); self.assertGreater(r["ci95"][0], 0.9)


if __name__ == "__main__":
    unittest.main()
