"""Numpy evaluator for an exported scikit-learn HistGradientBoostingClassifier (binary), so the research count
model B-P15 runs in the main environment without scikit-learn. Export: scripts/phase15/train_count15.py."""
from __future__ import annotations

import numpy as np


class Forest:
    def __init__(self, d: dict):
        self.base = float(d["baseline"]); self.trees = d["trees"]; self.mu = np.asarray(d["mu"]); self.sd = np.asarray(d["sd"])

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        return cls(json.loads(Path(path).read_text()))

    def raw(self, x: np.ndarray) -> float:
        z = (np.asarray(x, float) - self.mu) / self.sd
        s = self.base
        for t in self.trees:
            i = 0
            while not t["is_leaf"][i]:
                v = z[t["feature"][i]]
                go_left = (v <= t["thr"][i]) if np.isfinite(v) else t["missing_left"][i]
                i = t["left"][i] if go_left else t["right"][i]
            s += t["value"][i]
        return s

    def prob(self, x) -> float:
        return float(1.0 / (1.0 + np.exp(-self.raw(x))))
