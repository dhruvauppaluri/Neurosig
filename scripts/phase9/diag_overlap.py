import sys
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.pipeline import Phase8Localizer
from edge.phase9.configs import P9_CONDITIONER, p9_factory
from edge.phase9.improved import condition
from edge.phase9.synth_real import make_events, xy_error
ZM = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
array, ev = make_events("dev", n_per_noise=40, contaminated_fraction=0.5, z_max=ZM)
print("z_max", ZM)
w = p9_factory(array)["weighted_scan"]
for kw in (dict(model="line_source", lambda_um=20.0, rank=4),):
    loc = Phase8Localizer(array, **kw)
    e1, ew, d, ds = [], [], [], []
    for e in ev:
        sn = condition(e.snippet, **P9_CONDITIONER)
        f = loc.fitter.fit(sn)
        s = f.sources1[0]
        e1.append(np.hypot(s.x - e.truth_xy[0], s.y - e.truth_xy[1])); d.append(f.delta); ds.append(f.delta_shared)
        r = w.estimate(sn); ew.append(xy_error(r.xyz_um, e.truth_xy))
    c = np.array([e.contaminated for e in ev]); e1 = np.array(e1); ew = np.array(ew); d = np.array(d); ds = np.array(ds)
    print(kw, "K=1 VP median err single/pair", np.median(e1[~c]).round(1), np.median(e1[c]).round(1), "| weighted_scan", np.median(ew[~c]).round(1), np.median(ew[c]).round(1))
    print("   delta single pct50/90/99", np.percentile(d[~c], [50, 90, 99]).round(0), "pair pct10/50/90", np.percentile(d[c], [10, 50, 90]).round(0),
          "| delta_shared single", np.percentile(ds[~c], [50, 90, 99]).round(1), "pair", np.percentile(ds[c], [10, 50, 90]).round(1))
# chi2_red per noise
sn = condition(ev[0].snippet, **P9_CONDITIONER); f = Phase8Localizer(array, model="line_source", lambda_um=20.0, rank=None).fitter
Wt, sig = f.whiten(sn); print("whitened shape", Wt.shape, "sse1/size", f.fit(sn).sse1 / Wt.size)
