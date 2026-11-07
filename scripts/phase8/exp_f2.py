"""Reduced-rank fit: speed and detection/accuracy vs full rank on dev overlap events."""
import time
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase8.overlap_bench import assign, make_overlap_events
array = _array("linear")
ev = make_overlap_events("dev", n_per_noise=30)
res = np.array([e.resolvable for e in ev]); sg = np.array([e.kind == "single" for e in ev])
for rank in (None, 6, 4, 3, 2):
    f = SpaceTimeFitter(array, rank=rank)
    t = time.perf_counter(); fits = [f.fit(e.snippet) for e in ev]; ms = (time.perf_counter() - t) / len(ev) * 1000
    d = np.array([x.delta for x in fits]); ds = np.array([x.delta_shared for x in fits])
    tf = np.percentile(d[sg], 97); ts = np.percentile(ds[sg], 99)
    two = (d > tf) | (ds > ts)
    # single-source primary error
    err1 = []
    for e, x in zip(ev, fits):
        if e.kind == "single":
            t0 = np.array(e.truths_rho_y[0]); err1.append(np.hypot(x.sources1[0].x - t0[0], x.sources1[0].y - t0[1]))
    print(f"rank={rank}: {ms:.1f} ms/event tau_free={tf:.0f} tau_sh={ts:.1f} recall_res={two[res].mean():.3f} "
          f"false_single={two[sg].mean():.3f} single median err={np.median(err1):.1f}")
