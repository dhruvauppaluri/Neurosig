"""Prescreen: grid-level deltas vs refined; pick safe thresholds; measure speed/recall."""
import json, time
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase8.overlap_bench import make_overlap_events
array = _array("linear")
ev = make_overlap_events("dev", n_per_noise=30)
full = SpaceTimeFitter(array)
rows = []
for e in ev:
    f = full.fit(e.snippet)
    # grid-level via a prescreen fitter that never refines K=2 (huge thresholds)
    rows.append((f.delta, f.delta_shared))
rows = np.array(rows)
for pf, ps in ((0, 0), (20, 2), (40, 3), (60, 4), (80, 5)):
    fit = SpaceTimeFitter(array, prescreen_free=pf, prescreen_shared=ps)
    t = time.perf_counter(); out = [fit.fit(e.snippet) for e in ev]; dt = (time.perf_counter() - t) / len(ev) * 1000
    d = np.array([o.delta for o in out]); ds = np.array([o.delta_shared for o in out])
    keep_free = ((rows[:, 0] > 138) == (d > 138)).mean(); keep_sh = ((rows[:, 1] > 9) == (ds > 9)).mean()
    print(f"pre_free={pf} pre_sh={ps}: {dt:.1f} ms/event; decision agreement free {keep_free:.3f} shared {keep_sh:.3f}")
