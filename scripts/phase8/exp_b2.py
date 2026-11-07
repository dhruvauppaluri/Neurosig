"""Overlap benchmark on the dev pool: baselines vs phase8 pipeline (calibration from reports/phase8/calibration_v2.json)."""
import json, sys
import numpy as np
from edge.benchmark_v3 import _array, _engines
from edge.localize.common import fold_linear_probe_ambiguity
from edge.phase8.overlap_bench import evaluate_methods, make_overlap_events, summarize_overlap
from edge.phase8.pipeline import Calibration, Phase8Localizer

pool = sys.argv[1] if len(sys.argv) > 1 else "dev"
array = _array("linear")
events = make_overlap_events(pool, n_per_noise=30)
cal = Calibration(**json.load(open("reports/phase8/calibration_v2.json")))
p8 = Phase8Localizer(array, cal)
base = _engines(array)

def baseline(name):
    def fn(sn):
        e = base[name].estimate(sn)
        if not e.available:
            return np.zeros((0, 2)), True
        a = fold_linear_probe_ambiguity(e.xyz_um, sn.electrode_xyz_um)
        return np.array([[a[0], a[1]]]), False
    return fn

def phase8(sn):
    r = p8.localize(sn)
    if r.primary is None:
        return np.zeros((0, 2)), True
    est = np.array([s.xyz_um[:2] for s in r.sources])
    return est, (not r.accepted)

methods = {"beamform": baseline("beamform"), "template": baseline("template"), "robust_scan": baseline("robust_scan"),
           "phase8": phase8}
recs = evaluate_methods(events, methods)
for name, r in recs.items():
    s = summarize_overlap(events, r)
    print("==", name)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "n"}, indent=None))
print(summarize_overlap(events, recs["phase8"])["n"])
