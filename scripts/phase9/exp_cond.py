"""E2-E7: conditioner ablations on development windows, raw path, all five engines. One option per run."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.factories import baseline_factory
from edge.phase9.improved import conditioned_factory
S="/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/"
ev = H.build_events(H.manifest_windows("dev"), conditioning="raw")
variants = {"baseline": baseline_factory,
            "E2_baseline_sub": conditioned_factory(baseline_factory, baseline=True),
            "E3_snippet_car": conditioned_factory(baseline_factory, car=True),
            "E4_tmedian3": conditioned_factory(baseline_factory, tmedian=True),
            "E5_noise_norm": conditioned_factory(baseline_factory, noise_norm=True),
            "E6_interp_dead": conditioned_factory(baseline_factory, interp_dead=True)}
res = {}
for name, f in variants.items():
    t=time.time(); res[name] = H.run(ev, f); print(name, round(time.time()-t), "s", flush=True)
json.dump(res, open(S+"cond.json","w"), default=float)
def g(r, k, f="median"):
    return r.get(k, {}).get(f)
cols = ["drop_clustered_0.75","dropU_random_0.5","dropU_clustered_0.75","noise_independent","noise_impulsive","noise_gain","noise_drift"]
for m in ["amplitude","template","beamform","weighted_scan","robust_scan"]:
    print("==", m)
    print("variant".ljust(18), "valid  ax_med  spread  rho   w/b  ", *[c.replace("noise_","n_").replace("dropU_","U_").replace("drop_","D_")[:14].ljust(9) for c in cols])
    for v, r in res.items():
        x = r[m]
        print(v.ljust(18), f"{x['valid_fraction']:.2f}  {x['axial_disagreement_um']['median']:6.1f}  {x['within_unit_spread_um']['median']:6.1f}  {x['unit_depth_rank_corr_median']:.2f}  {x['within_over_between_unit_spread']:.2f}  ",
              *[f"{g(x,c):6.1f}   " for c in cols])
