"""Phase 15 research count configuration B-P15: gradient-boosted trees (edge/phase15/trees.py) on the spatially whitened
line20 and ms3 model-comparison statistics plus temporal features, with the same per-recording label-free threshold
self-calibration as P14 (quantile from reports/phase15/count_b_p15.json of development-morphology singles placed in
the recording's own background). Feature definitions match scripts/phase15/overlap_lomo.py (FAM = line20, ms3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edge.phase14.neuron_model import SourceModel, prepare
from edge.phase15.trees import Forest

ROOT = Path(__file__).resolve().parents[2]


class CountB:
    def __init__(self, noise, path=ROOT / "reports/phase15/count_b_p15.json"):
        d = json.loads(Path(path).read_text())
        self.forest = Forest(d); self.q = d["quantile"]; self.noise = noise; self._m = {}; self._loc = {}
        self.thr = np.inf

    def _models(self, E):
        k = E.tobytes()
        if k not in self._m:
            from edge.phase12.fast import fast_localizer
            from edge.phase12.overlap_bench import _array
            self._m[k] = ({s: SourceModel(E, s) for s in ("line20", "ms3")}, fast_localizer(_array(E)))
        return self._m[k]

    def features(self, sn) -> np.ndarray:
        from edge.phase12 import count_model as CM
        E = np.asarray(sn.electrode_xyz_um, float)
        mods, loc = self._models(E)
        a = CM.analyze_fast(loc, sn) or {}
        snr = float(np.max(-prepare(sn).D.min(axis=1)))
        pr = prepare(sn, noise=self.noise, temporal=False)
        f = []
        fits = {s: m.fit(pr, 3) for s, m in mods.items()}
        for s in ("line20", "ms3"):
            y = fits[s]; f += [np.log1p(max(y["F21"], 0)), np.log(max(y["chi1"], 1e-6))]
        m = fits["ms3"]
        f += [m["c1_pol"], np.log1p(m["c1_trough"]), np.log(max(m["en_ratio"], 1e-6)), min(m["sep"], 200) / 60, min(m["dt_comp"], 30) / 10,
              np.log1p(max(m.get("F32", 0), 0)), a.get("rank2", 0.0), min(a.get("n_trough", 0), 4), min(a.get("t_spread", 0.0), 10.0), np.log1p(snr)]
        return np.nan_to_num(np.array(f, float), nan=0.0, posinf=50, neginf=-50)

    def score(self, sn) -> float:
        return self.forest.prob(self.features(sn))

    def calibrate(self, cal_snippets):
        self.thr = float(np.quantile([self.score(s) for s in cal_snippets], self.q))
        return self.thr
