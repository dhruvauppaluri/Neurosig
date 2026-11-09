"""Phase 14 regression tests: single-neuron null models and count statistic, fast/slow equivalence, noise whitening,
online unit association, drift tracking, Allen electrode-to-waveform mapping, split integrity, determinism.
Tests needing git-ignored data files are skipped when the files are absent."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from edge.types import Snippet
from edge.phase14 import neuron_model as N
from edge.phase14.allen_map import bin_counts, map_waveform_rows
from edge.phase14.fast14 import FastSourceModel
from edge.phase14.online_units import OnlineUnits, Params, ari, evaluate
from edge.phase14 import drift_sim as DS

ROOT = Path(__file__).resolve().parents[2]


def _geom():
    rows = np.arange(16) * 20.0
    E = []
    for i, y in enumerate(rows):
        for x in ((11.0, 43.0) if i % 2 == 0 else (27.0, 59.0)):
            E.append((x, y))
    return np.array(E)


def _event(E, centres, rng, noise=1.0, dt=(0,), amp=(30.0,)):
    """Synthetic multichannel event from the ms3 basis at given centres (one temporal waveform per basis column)."""
    T = 100
    t = np.arange(T) - 40.0
    v = np.zeros((T, len(E)))
    for c, d, a in zip(centres, dt, amp):
        K = N.kernel_cols(np.array([c]), E, N.SPECS["ms3"])[0]       # [C, 3]
        h0 = -np.exp(-0.5 * ((t - d) / 3.0) ** 2)
        h1 = 0.3 * np.exp(-0.5 * ((t - d - 6) / 5.0) ** 2)
        h2 = 0.1 * np.gradient(h0)
        v += a * (np.outer(h0, K[:, 0]) + np.outer(h1, K[:, 1]) + np.outer(h2, K[:, 2]))
    v += noise * rng.standard_normal(v.shape)
    E3 = np.c_[E, np.zeros(len(E))]
    return Snippet(0, 40, 0, 30000.0, tuple(range(len(E))), v, E3, None, None)


class TestSourceModels(unittest.TestCase):
    def setUp(self):
        self.E = _geom(); self.rng = np.random.default_rng(1)

    def test_two_sources_raise_F21(self):
        m = N.SourceModel(self.E, "ms3")
        f1 = [m.fit(N.prepare(_event(self.E, [(30, 150)], self.rng)), 2)["F21"] for _ in range(4)]
        f2 = [m.fit(N.prepare(_event(self.E, [(30, 90), (30, 210)], self.rng, dt=(0, 4), amp=(30, 25))), 2)["F21"] for _ in range(4)]
        self.assertGreater(min(f2), max(f1))

    def test_single_source_position(self):
        m = N.SourceModel(self.E, "ms3")
        f = m.fit(N.prepare(_event(self.E, [(30, 150)], self.rng, noise=0.2)), 1)
        self.assertLess(np.linalg.norm(f["p1"] - np.array([30, 150])), 6.0)

    def test_fast_equals_reference(self):
        slow, fast = N.SourceModel(self.E, "ms3"), FastSourceModel(self.E, "ms3", n_cand=4, rounds=2)
        for k in range(4):
            sn = _event(self.E, [(30, 90), (40, 200)] if k % 2 else [(30, 150)], self.rng, dt=(0, 3), amp=(30, 20))
            pr = N.prepare(sn)
            a, b = slow.fit(pr, 2), fast.fit(pr, 2)
            self.assertLess(abs(a["F21"] - b["F21"]) / a["F21"], 0.05)
            self.assertLess(np.linalg.norm(a["p1"] - b["p1"]), 5.0)

    def test_fast_deterministic(self):
        fast = FastSourceModel(self.E, "ms3")
        sn = _event(self.E, [(30, 150)], self.rng)
        pr = N.prepare(sn)
        self.assertEqual(fast.fit(pr, 3)["F21"], fast.fit(pr, 3)["F21"])

    def test_noise_model_whitens(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((32, 32)) * 0.3 + np.eye(32)
        segs = rng.standard_normal((800, 120, 32)) @ A.T
        nm = N.NoiseModel(segs)
        live = np.ones(32, bool)
        Ws, _ = nm.spatial(live)
        S = Ws @ nm.Ss @ Ws.T
        self.assertTrue(np.allclose(S, np.eye(32), atol=1e-6))


class TestOnlineUnits(unittest.TestCase):
    def test_two_units_recovered(self):
        rng = np.random.default_rng(3)
        E = _geom()
        protos = []
        for ch in (6, 24):
            w = np.zeros((45, 32)); w[15, ch] = -12; w[16, ch] = -6; w[17, ch] = 3
            for nb in np.flatnonzero(np.linalg.norm(E - E[ch], axis=1) <= 30):
                if nb != ch:
                    w[15, nb] = -5
            protos.append((w, ch))
        ou = OnlineUnits(E, Params(min_gain=20, merge_every=10 ** 9, snr_assign=0.0))
        lab = []
        for i in range(300):
            u = int(rng.integers(2)); w, ch = protos[u]
            ou.add(w + rng.standard_normal(w.shape), np.zeros(32), ch, i * 0.01, 12.0, i); lab.append(u)
        ev = evaluate(ou.final_labels(), np.array(lab), np.full(300, 12.0))
        self.assertGreaterEqual(ev["precision"], 0.99)
        self.assertGreater(ev["coverage_all"], 0.8)
        self.assertLessEqual(ev["fragmentation_mean"], 1.01)

    def test_ari(self):
        a = np.array([0, 0, 1, 1, 2, 2]); self.assertAlmostEqual(ari(a, a), 1.0)
        self.assertLess(ari(a, np.array([0, 1, 0, 1, 0, 1])), 0.1)

    def test_kalman_tracks_drift(self):
        t = np.linspace(0, 100, 500); truth = np.c_[np.zeros(500), 0.3 * t]
        P = truth + np.random.default_rng(4).normal(0, 10, truth.shape)
        st = DS.track_positions(t, P, "static"); ka = DS.track_positions(t, P, "kalman", q=2.0, r=15.0)
        self.assertLess(np.median(np.abs(ka[100:, 1] - truth[100:, 1])), np.median(np.abs(st[100:, 1] - truth[100:, 1])))


class TestAllenMapping(unittest.TestCase):
    def test_rows_follow_local_index_not_table_order(self):
        """Regression for the Phase 13 defect: electrodes table order != local_index order."""
        rng = np.random.default_rng(5)
        n = 384
        li = rng.permutation(n)                                  # table row k has local_index li[k]
        probe = np.zeros(n, int)
        peak = 17
        rows = map_waveform_rows(li, probe, peak)
        self.assertEqual(len(rows), n)
        for r in range(n):
            self.assertEqual(li[rows[r]], r)
        # a waveform whose trough is on row li[peak] must map back to the peak electrode
        self.assertEqual(rows[li[peak]], peak)

    def test_two_probes(self):
        li = np.r_[np.arange(4)[::-1], np.arange(4)]; probe = np.r_[np.zeros(4, int), np.ones(4, int)]
        self.assertTrue(np.array_equal(map_waveform_rows(li, probe, 5), np.array([4, 5, 6, 7])))
        self.assertTrue(np.array_equal(map_waveform_rows(li, probe, 0), np.array([3, 2, 1, 0])))

    def test_bin_counts(self):
        rng = np.random.default_rng(6)
        on = np.sort(rng.uniform(0, 100, 50)); on = on[np.r_[True, np.diff(on) > 0.3]]
        sp = np.sort(rng.uniform(0, 100, 3000))
        ref = np.array([np.histogram(sp - o, np.arange(51) * 0.005)[0] for o in on])
        self.assertTrue(np.array_equal(bin_counts(sp, on), ref))


class TestSplits(unittest.TestCase):
    def test_morphology_disjoint(self):
        from edge.phase14.bench14 import DEV_MODELS, SEL_MODELS
        import importlib.util
        spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/phase14/gen_templates14.py")
        g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
        self.assertFalse(set(DEV_MODELS) & set(SEL_MODELS))
        self.assertFalse(set(g.UNSEEN) & (set(DEV_MODELS) | set(SEL_MODELS)))
        p11 = {"L5_TTPC1_cADpyr232_1", "L5_UTPC_cADpyr232_1", "L5_STPC_cADpyr232_1", "L5_LBC_bAC217_1", "L5_NBC_bAC217_1", "L5_MC_bAC217_1"}
        self.assertFalse(set(g.UNSEEN) & p11)

    def test_ibl_final_sessions_fresh(self):
        p = ROOT / "reports/phase14/split_manifest_phase14.json"
        if not p.exists():
            self.skipTest("manifest missing")
        fresh = {v["pid"] for v in json.loads(p.read_text())["sessions"].values()}
        for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json", "reports/phase10/split_manifest_phase10.json",
                  "reports/phase11/split_manifest_phase11.json", "reports/phase12/split_manifest_phase12.json"):
            q = ROOT / f
            if q.exists():
                old = {v["pid"] for v in json.loads(q.read_text())["sessions"].values()}
                self.assertFalse(fresh & old, f)

    def test_allen_final_mice_disjoint(self):
        p = ROOT / "reports/phase14/allen_split14.json"
        if not p.exists():
            self.skipTest("split missing")
        s = json.loads(p.read_text())
        dev_sel = set(s["dev"]) | set(s["select_p13"]) | set(s["train_candidates"])
        self.assertFalse(set(s["final_candidates"]) & dev_sel)

    def test_frozen_hash(self):
        p = ROOT / "reports/phase14/frozen_p14.json"
        if not p.exists():
            self.skipTest("frozen config missing")
        d = json.loads(p.read_text()); h = d.pop("hash")
        self.assertEqual(hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12], h)
        self.assertEqual(h, "9d8006e97d8f")


if __name__ == "__main__":
    unittest.main()
