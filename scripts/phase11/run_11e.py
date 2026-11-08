"""Phase 11E: calibrate count rules on p11dev, compare on p11select (development decision pool).
Rules: P10 (frozen Phase 9 calibration, dead contacts masked), OV-A, OV-A+B, OV-A+B+K3.
Calibration (p11dev only): tau_two_norm = 97th pct of normalized delta on singles; tau_shared_norm = 99th pct of
normalized delta_shared on singles; tau_three = 97th pct of normalized delta3 on resolvable pairs declared two."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase10.overlap_stress import localizer, make_stress, summarize
from edge.phase11.overlap import CountRule, evaluate

def pool(name, n):
    p = Path(f"data/phase11/stress_{name}.pkl")
    if p.exists():
        return pickle.load(open(p, "rb"))
    r = make_stress(name, n_per_cell=n); pickle.dump(r, open(p, "wb")); return r

array, dev = pool("p11dev", 30)
loc = localizer(array)
base = CountRule()
A = CountRule(axial_only_boundary=True)
# calibration statistics under OV-A with normalization
probe = CountRule(axial_only_boundary=True, normalized=True, tau_two=np.inf, tau_shared=np.inf)
ds = [evaluate(loc, e, probe) for e in dev["single"]]
tau2 = float(np.percentile([r["delta_norm"] for r in ds], 97)); taus = float(np.percentile([r["delta_shared_norm"] for r in ds], 99))
AB = CountRule(axial_only_boundary=True, normalized=True, tau_two=tau2, tau_shared=taus)
pr = [evaluate(loc, e, CountRule(axial_only_boundary=True, normalized=True, tau_two=tau2, tau_shared=taus, tau_three=-np.inf))
      for e in dev["dt"] + dev["sep"] if e.resolvable]
d3 = [r["delta3_norm"] for r in pr if r["delta3_norm"] is not None]
tau3 = float(np.percentile(d3, 97))
ABK = CountRule(axial_only_boundary=True, normalized=True, tau_two=tau2, tau_shared=taus, tau_three=tau3)
cal = {"tau_two_norm": tau2, "tau_shared_norm": taus, "tau_three_norm": tau3}
print("calibration", cal, flush=True)
rules = {"P10": base, "OV-A": A, "OV-A+B": AB, "OV-A+B+K3": ABK}

def report(evs, rule):
    rows = {f: [evaluate(loc, e, rule) for e in evs[f]] for f in evs}
    pairs = rows["dt"] + rows["sep"]
    res = [r for r in pairs if r["res"]]
    o = {"resolvable_pairs": summarize(res, 2), "all_pairs": summarize(pairs, 2), "singles": summarize(rows["single"], 1),
         "false_two_singles": float(np.mean([r["n_est"] >= 2 for r in rows["single"]])),
         "false_three_resolvable_pairs": float(np.mean([r["n_est"] >= 3 for r in res])),
         "simultaneous_resolvable": summarize([r for r in res if r["dt"] == 0], 2),
         "simultaneous_all": summarize([r for r in pairs if r["dt"] == 0], 2),
         "triples": summarize(rows["triple"], 3), "triples_resolvable": summarize([r for r in rows["triple"] if r["res"]], 3),
         "rows": rows}
    return o

out = {"calibration": cal, "dev": {}, "select": {}}
array, sel = pool("p11select", 20)
for name, rule in rules.items():
    for split, evs in (("dev", dev), ("select", sel)):
        o = report(evs, rule)
        rows = o.pop("rows")
        out[split][name] = o
        if split == "select":
            pickle.dump(rows, open(f"data/phase11/rows11e_{name}.pkl", "wb"))
        print(f"{split:6s} {name:10s} res-recall {o['resolvable_pairs']['all_sources_recovered']:.3f} all-pairs {o['all_pairs']['all_sources_recovered']:.3f} "
              f"false2 {o['false_two_singles']:.3f} simult {o['simultaneous_resolvable']['all_sources_recovered']:.3f} "
              f"err {o['resolvable_pairs']['matched_err_median_um']:.1f} triples {o['triples']['all_sources_recovered']:.3f} "
              f"false3 {o['false_three_resolvable_pairs']:.3f} acc-res {o['resolvable_pairs']['accepted']:.3f}", flush=True)
Path("reports/phase11/exp11e_overlap.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
