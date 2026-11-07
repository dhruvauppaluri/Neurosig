"""Why are resolvable pairs missed? delta vs noise/dt on dev overlap events; sweep tau_two."""
import json
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.overlap_bench import make_overlap_events
from edge.phase8.pipeline import Calibration, Phase8Localizer

array = _array("linear")
ev = make_overlap_events("dev", n_per_noise=30)
cal = Calibration(**json.load(open("reports/phase8/calibration_v2.json")))
p8 = Phase8Localizer(array, cal)
d = np.array([p8.fitter.fit(e.snippet).delta for e in ev])
res = np.array([e.resolvable for e in ev]); sg = np.array([e.kind == "single" for e in ev])
noise = np.array([e.noise_uv for e in ev]); dt = np.array([abs(e.dt_samples[1]) if len(e.dt_samples) > 1 else -1 for e in ev])
print("current tau", cal.tau_two)
for nz in (2.0, 5.0, 10.0, 25.0):
    m = res & (noise == nz)
    print(nz, "resolvable n", m.sum(), "delta median", np.median(d[m]).round(0) if m.any() else None,
          "recall@tau", (d[m] > cal.tau_two).mean().round(2) if m.any() else None,
          "| single delta p95", np.percentile(d[sg & (noise == nz)], 95).round(0))
for lo, hi in ((0, 2), (3, 6), (7, 12)):
    m = res & (dt >= lo) & (dt <= hi)
    print("dt", lo, hi, "n", m.sum(), "recall", (d[m] > cal.tau_two).mean().round(2))
for tau in (100, 110, 120, 130, 138, 150):
    print(tau, "recall_res", (d[res] > tau).mean().round(3), "false_single", (d[sg] > tau).mean().round(3))
