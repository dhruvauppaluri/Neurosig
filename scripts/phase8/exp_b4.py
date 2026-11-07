import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase8.overlap_bench import make_overlap_events
array = _array("linear"); f = SpaceTimeFitter(array)
ev = make_overlap_events("dev", n_per_noise=30)
fits = [f.fit(e.snippet) for e in ev]
d = np.array([x.delta for x in fits]); ds = np.array([x.delta_shared for x in fits])
res = np.array([e.resolvable for e in ev]); sg = np.array([e.kind == "single" for e in ev])
dt = np.array([abs(e.dt_samples[1]) if len(e.dt_samples) > 1 else -1 for e in ev])
print("single delta_shared pctl 50/90/95/97/99:", np.percentile(ds[sg], [50, 90, 95, 97, 99]).round(1))
for lo, hi in ((0, 2), (3, 6), (7, 12)):
    m = res & (dt >= lo) & (dt <= hi); print("dt", lo, hi, "n", m.sum(), "delta_shared median", np.median(ds[m]).round(1), "p10", np.percentile(ds[m], 10).round(1))
for tsh in (10, 15, 20, 30):
    print(tsh, "recall_res(either @138)", np.mean((d[res] > 138) | (ds[res] > tsh)).round(3),
          "false_single", np.mean((d[sg] > 138) | (ds[sg] > tsh)).round(3))
