import json
import numpy as np
from edge.benchmark_v3 import _array
from edge.phase8.overlap_bench import make_overlap_events, assign
from edge.phase8.pipeline import Calibration, Phase8Localizer
array = _array("linear")
cal = Calibration(**json.load(open("reports/phase8/calibration_v2.json")))
p8 = Phase8Localizer(array, cal)
ev = make_overlap_events("dev", n_per_noise=30)
for e in ev:
    if not e.resolvable: continue
    r = p8.localize(e.snippet)
    a = assign(np.asarray(e.truths_rho_y), np.array([s.xyz_um[:2] for s in r.sources]))
    if not (r.n_sources == 2 and a["matched"] == 2):
        fit = p8.fitter.fit(e.snippet)
        print(f"noise {e.noise_uv} dt {e.dt_samples[1]} sep {e.sep_um:.0f} ratio {e.ratio:.2f} snr {np.round(e.snr,1)} "
              f"n_est {r.n_sources} flags {r.flags} delta {fit.delta:.0f} dsh {fit.delta_shared:.1f} matched {a['matched']} "
              f"truth {np.round(e.truths_rho_y,0).tolist()} est {[np.round(s.xyz_um[:2],0).tolist() for s in r.sources]}")
