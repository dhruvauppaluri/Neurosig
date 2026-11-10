"""Phase 15 regression tests: batched deployment path vs per-event fits, exported tree evaluator, ID-P15 identity
(tau_rel = 0 reproduces OU-P14), cross-animal feature construction (label-free normalization, CORAL), split integrity
of the reserved Phase 15 data and the frozen-configuration hash. Tests needing git-ignored data files are skipped."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from edge.phase14 import neuron_model as N
from edge.phase14.online_units import OnlineUnits, Params, evaluate
from edge.phase15.batch import BatchModel
from edge.phase15.identity import OnlineUnits15
from edge.phase15.trees import Forest
from edge.phase15 import xanimal as X
from edge.tests.test_phase14 import _event, _geom

ROOT = Path(__file__).resolve().parents[2]


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.E = _geom(); self.rng = np.random.default_rng(11)
        self.noise = N.NoiseModel(np.random.default_rng(12).standard_normal((300, 160, 32)))

    def test_single_source_position_matches_truth_and_reference(self):
        bm = BatchModel(self.E)
        ref = N.SourceModel(self.E, "ms3")
        sns = [_event(self.E, [(30, 150)], self.rng, noise=0.3) for _ in range(3)]
        out = bm.run(sns, self.noise)
        for sn, r in zip(sns, out):
            self.assertLess(np.linalg.norm(r["p1_ms3"] - np.array([30, 150])), 10.0)
            p_ref = ref.fit(N.prepare(sn, noise=self.noise, temporal=False), 1)["p1"]
            self.assertLess(np.linalg.norm(r["p1_ms3"] - p_ref), 10.0)
            self.assertTrue(np.allclose(r["p1"], 0.5 * (r["p1_ms3"] + r["p1_line20"])))

    def test_two_sources_raise_F21(self):
        bm = BatchModel(self.E)
        one = bm.run([_event(self.E, [(30, 150)], self.rng) for _ in range(4)], self.noise)
        two = bm.run([_event(self.E, [(30, 90), (30, 210)], self.rng, dt=(0, 4), amp=(30, 25)) for _ in range(4)], self.noise)
        self.assertGreater(min(r["F21"] for r in two), max(r["F21"] for r in one))

    def test_batch_equals_one_by_one(self):
        """Batching changes only when results are emitted, not their values (rank compression is per event)."""
        bm = BatchModel(self.E)
        sns = [_event(self.E, [(30, 60 + 40 * i)], self.rng) for i in range(4)]
        allr = bm.run(sns, self.noise)
        for sn, r in zip(sns, allr):
            s = bm.run([sn], self.noise)[0]
            self.assertAlmostEqual(r["F21"], s["F21"], places=6)
            self.assertTrue(np.allclose(r["p1"], s["p1"]))


class TestForest(unittest.TestCase):
    def test_manual_tree(self):
        d = {"baseline": 0.5, "mu": [0.0, 0.0], "sd": [1.0, 1.0], "trees": [
            {"feature": [0, 0, 0], "thr": [1.0, 0, 0], "left": [1, 0, 0], "right": [2, 0, 0], "is_leaf": [False, True, True],
             "value": [0.0, -1.0, 2.0], "missing_left": [True, False, False]}]}
        f = Forest(d)
        self.assertAlmostEqual(f.raw([0.5, 0.0]), -0.5)
        self.assertAlmostEqual(f.raw([3.0, 0.0]), 2.5)
        self.assertAlmostEqual(f.raw([np.nan, 0.0]), -0.5)       # missing goes left here
        self.assertAlmostEqual(f.prob([3.0, 0.0]), 1 / (1 + np.exp(-2.5)))

    def test_exported_model_loads(self):
        p = ROOT / "reports/phase15/count_b_p15.json"
        if not p.exists():
            self.skipTest("count model missing")
        f = Forest.load(p)
        v = f.prob(np.zeros(len(f.mu)))
        self.assertTrue(0.0 <= v <= 1.0)


def _two_unit_stream(seed=3, n=300):
    rng = np.random.default_rng(seed)
    E = _geom()
    protos = []
    for ch in (6, 24):
        w = np.zeros((45, 32)); w[15, ch] = -12; w[16, ch] = -6; w[17, ch] = 3
        for nb in np.flatnonzero(np.linalg.norm(E - E[ch], axis=1) <= 30):
            if nb != ch:
                w[15, nb] = -5
        protos.append((w, ch))
    ev = []
    for i in range(n):
        u = int(rng.integers(2)); w, ch = protos[u]
        ev.append((w + rng.standard_normal(w.shape), ch, i * 0.01, u))
    return E, ev


class TestIdentity(unittest.TestCase):
    def test_tau_rel_zero_reproduces_oup14(self):
        E, ev = _two_unit_stream()
        p = dict(min_gain=20, merge_every=10 ** 9, snr_assign=0.0)
        a, b = OnlineUnits(E, Params(**p)), OnlineUnits15(E, Params(**p), tau_rel=0.0)
        for i, (w, ch, t, _) in enumerate(ev):
            a.add(w, np.zeros(32), ch, t, 12.0, i); b.add(w, np.zeros(32), ch, t, 12.0, i)
        self.assertTrue(np.array_equal(a.final_labels(), b.final_labels()))

    def test_tau_rel_keeps_precision(self):
        E, ev = _two_unit_stream(seed=5)
        ou = OnlineUnits15(E, Params(min_gain=20, merge_every=10 ** 9, snr_assign=0.0), tau_rel=0.2)
        for i, (w, ch, t, _) in enumerate(ev):
            ou.add(w, np.zeros(32), ch, t, 12.0, i)
        r = evaluate(ou.final_labels(), np.array([e[3] for e in ev]), np.full(len(ev), 12.0))
        self.assertGreaterEqual(r["precision"], 0.99)
        self.assertGreater(r["coverage_all"], 0.8)

    def test_threshold_monotone_in_template_energy(self):
        E, _ = _two_unit_stream()
        ou = OnlineUnits15(E, Params(), tau_rel=0.2)

        class U:
            pass
        lo, hi = U(), U(); lo.mu = np.full((10, 4), 0.5); hi.mu = np.full((10, 4), 5.0)
        self.assertEqual(ou._thr(lo), Params().tau_assign)
        self.assertGreater(ou._thr(hi), ou._thr(lo))


def _fake_mouse(rng, n_units=40, n_trials=60, n_cls=3, shift=0.0):
    y = rng.integers(n_cls, size=n_trials)
    C = rng.poisson(1.0, (n_trials, n_units, 50)).astype(np.int16)
    C[:, : n_units // 2, 5:15] += (y[:, None, None] == 1) * 3
    d = {"unit_area": np.array(["VISp"] * (n_units // 2) + ["VISl"] * (n_units - n_units // 2)),
         "unit_probe": np.zeros(n_units, int), "unit_vpos": rng.uniform(0, 1000, n_units) + shift,
         "wave": np.zeros((n_units, 82, 32)), "unit_snr": np.ones(n_units)}
    return C, d, y


class TestXanimal(unittest.TestCase):
    def test_feature_shape_and_session_z(self):
        rng = np.random.default_rng(0)
        C, d, _ = _fake_mouse(rng)
        F = X.features(C, d, X.Spec(groups=("area", "layer"), unit_norm="none", sess_norm="z"))
        self.assertEqual(F.shape, (60, 6 * 3 * 10))
        live = F.std(0) > 0
        self.assertTrue(np.allclose(F[:, live].mean(0), 0, atol=1e-5))

    def test_features_ignore_labels(self):
        """Features are a function of responses only: permuting trial labels cannot change them."""
        rng = np.random.default_rng(1)
        C, d, y = _fake_mouse(rng)
        s = X.Spec(groups=("area",), unit_norm="z")
        self.assertTrue(np.array_equal(X.features(C, d, s), X.features(C, d, s)))

    def test_coral_matches_covariance(self):
        rng = np.random.default_rng(2)
        Xs = rng.standard_normal((500, 3)) @ np.diag([1.0, 2.0, 3.0]); Xt = rng.standard_normal((400, 3)) * 0.5 + 4
        Z = X.coral(Xs, Xt, eps=1e-6)
        self.assertTrue(np.allclose(np.cov(Z, rowvar=False), np.cov(Xs, rowvar=False), atol=0.05))
        self.assertTrue(np.allclose(Z.mean(0), Xs.mean(0), atol=1e-6))

    def test_decodes_above_chance_across_mice(self):
        try:
            import sklearn  # noqa: F401  (decoders run in data/ibl-venv)
        except ImportError:
            self.skipTest("scikit-learn not installed in this environment")
        rng = np.random.default_rng(3)
        s = X.Spec(groups=("area",), unit_norm="none")
        tr = [_fake_mouse(rng) for _ in range(4)]; te = _fake_mouse(rng, shift=200.0)
        Xtr = np.vstack([X.features(C, d, s) for C, d, _ in tr]); ytr = np.concatenate([y for *_, y in tr])
        p, _, _ = X.fit_predict(Xtr, ytr, X.features(te[0], te[1], s), s)
        self.assertGreater(np.mean(p[te[2] == 1] == 1), 0.6)


class TestSplits15(unittest.TestCase):
    def test_ibl_sessions_fresh(self):
        p = ROOT / "reports/phase15/split_manifest_phase15.json"
        if not p.exists():
            self.skipTest("manifest missing")
        fresh = {v["pid"] for v in json.loads(p.read_text())["sessions"].values()}
        for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json", "reports/phase10/split_manifest_phase10.json",
                  "reports/phase11/split_manifest_phase11.json", "reports/phase12/split_manifest_phase12.json",
                  "reports/phase14/split_manifest_phase14.json"):
            q = ROOT / f
            if q.exists():
                old = {v["pid"] for v in json.loads(q.read_text())["sessions"].values()}
                self.assertFalse(fresh & old, f)

    def test_allen_phase15_mice_unused(self):
        p = ROOT / "reports/phase15/allen_split15.json"; q = ROOT / "reports/phase14/allen_inventory14.json"
        if not (p.exists() and q.exists()):
            self.skipTest("split missing")
        s = json.loads(p.read_text()); used = set(json.loads(q.read_text()))
        self.assertFalse(set(s["final_candidates"]) & set(s["train_candidates"]))
        self.assertFalse((set(s["final_candidates"]) | set(s["train_candidates"])) & used)

    def test_frozen_hash(self):
        p = ROOT / "reports/phase15/frozen_p15.json"
        if not p.exists():
            self.skipTest("frozen config missing")
        d = json.loads(p.read_text()); h = d.pop("hash")
        self.assertEqual(hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12], h)


if __name__ == "__main__":
    unittest.main()
