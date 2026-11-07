"""K-series: forward-model calibration x conditioners, on real dev (stability/null) and real-like synthetic dev (truth)."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.evalset import build_events_shift, evaluate, table
from edge.phase9.factories import baseline_factory, calibrated_factory
from edge.phase9.improved import condition, conditioned_factory
from edge.phase9.synth_real import make_events, xy_error
S="/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/"
wins = H.manifest_windows("dev")
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
array, syn = make_events("dev", n_per_noise=70)
B = dict(baseline=True); C = dict(baseline=True, car=True, interp_dead=True)
facs = {"legacy lam80/B": conditioned_factory(baseline_factory, **B),
        "lam80/C": conditioned_factory(baseline_factory, **C),
        "calib20/B": conditioned_factory(calibrated_factory(20, 30), **B),
        "calib20/C": conditioned_factory(calibrated_factory(20, 30), **C),
        "calib30/C": conditioned_factory(calibrated_factory(30, 40), **C),
        "calib15/C": conditioned_factory(calibrated_factory(15, 20), **C)}
res = {k: evaluate(f, real, null, dropout=True, noise=False) for k, f in facs.items()}
print(table(res, cols=("drop_clustered_0.75", "dropU_random_0.5", "dropU_clustered_0.75")))
print("\n=== real-like synthetic dev (n=%d), xy error um, accepted only" % len(syn))
syn_res = {}
for k, f in facs.items():
    eng = f(array); syn_res[k] = {}
    for m, e in eng.items():
        errs = [xy_error(r.xyz_um, ev.truth_xy) if (r := e.estimate(ev.snippet)).available else np.nan for ev in syn]
        a = np.array(errs); ok = np.isfinite(a)
        syn_res[k][m] = dict(acc=float(ok.mean()), med=float(np.median(a[ok])), p90=float(np.percentile(a[ok], 90)), large=float(np.mean(a[ok] > 50)))
    print(k.ljust(16), {m: f"{v['med']:.1f}/{v['p90']:.1f}/{100*v['acc']:.0f}%" for m, v in syn_res[k].items()})
json.dump({"real": res, "synth": syn_res}, open(S+"kseries.json","w"), default=float)
