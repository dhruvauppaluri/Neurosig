import sys
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.pipeline import Phase8Localizer
from edge.phase9.improved import condition
from edge.phase9.synth_real import make_events, xy_error
array, ev = make_events("dev", n_per_noise=40, contaminated_fraction=0.5)
c = np.array([e.contaminated for e in ev]); res = np.array([e.resolvable for e in ev])
for label, cond, kw in (("CAR, no nuisance", dict(baseline=True, car=True, interp_dead=True), dict()),
                        ("no CAR, nuisance CM", dict(baseline=True, interp_dead=True), dict(nuisance_cm=True)),
                        ("CAR + nuisance CM", dict(baseline=True, car=True, interp_dead=True), dict(nuisance_cm=True))):
    loc = Phase8Localizer(array, model="line_source", lambda_um=20.0, rank=4, **kw)
    d, ds, e1 = [], [], []
    for e in ev:
        f = loc.fitter.fit(condition(e.snippet, **cond))
        s = f.sources1[0]; d.append(f.delta); ds.append(f.delta_shared); e1.append(np.hypot(s.x - e.truth_xy[0], s.y - e.truth_xy[1]))
    d, ds, e1 = map(np.array, (d, ds, e1))
    t2 = np.percentile(d[~c], 97)
    print(f"{label:20s} K1 err single/pair {np.median(e1[~c]):.1f}/{np.median(e1[c]):.1f} | delta single p50/90/97 {np.percentile(d[~c],[50,90,97]).round(0)} resolvable p10/50 {np.percentile(d[res],[10,50]).round(0)} | recall@tau97 res {np.mean(d[res]>t2):.2f} all-pairs {np.mean(d[c]>t2):.2f}")
