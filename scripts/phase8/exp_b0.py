"""Diagnose K=1 VP fit and the K=2 delta statistic on dev events."""
import numpy as np
from edge.benchmark_v3 import _array, _engines
from edge.phase8.evaluate import fmt, run_engines, summarize
from edge.phase8.multisource import SpaceTimeFitter, SpaceTimeLocalizer
from edge.phase8.refine import RefinedScanLocalizer
from edge.phase8.synth import make_events

array = _array("linear")
events = make_events("dev", n_per_cell=40)
eng = {"refined": RefinedScanLocalizer(array), "vp1": SpaceTimeLocalizer(array)}
res = run_engines(eng, events)
S = {"single": lambda e: not e.contaminated, "contam": lambda e: e.contaminated,
     "noise25": lambda e: e.noise_uv == 25.0}
t = summarize(res, events, S)
for s in ("all", "single", "contam", "noise25"):
    print("---", s); print(fmt(t, s))
f = SpaceTimeFitter(array)
d = np.array([f.fit(e.snippet).delta for e in events])
c = np.array([e.contaminated for e in events]); n = np.array([e.noise_uv for e in events])
for nz in (2.0, 10.0, 25.0):
    m = n == nz
    print(nz, "delta single pctl 50/90/95/100:", np.percentile(d[m & ~c], [50, 90, 95, 100]).round(0),
          "contam pctl 5/25/50:", np.percentile(d[m & c], [5, 25, 50]).round(0))
