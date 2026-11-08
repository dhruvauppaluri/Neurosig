"""Phase 10 overlap stress run. K=3 threshold is calibrated on the p10dev pool only (false-three rate on accepted
correctly-resolved pairs <= 3%); everything is then reported on the p10test pool (single run, written once)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9.harness import make_mask
from edge.phase10.overlap_stress import evaluate_event, frozen_calibration, localizer, make_stress, summarize

OUT = Path("reports/phase10/overlap_stress.json")
if OUT.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{OUT} exists (test pool is single-use)")
t = time.time()
array, dev = make_stress("p10dev", n_per_cell=12)
loc = localizer(array)
d3 = []
for ev in dev["dt"] + dev["sep"]:
    r = evaluate_event(loc, ev, k3_tau=-np.inf)
    if r["res"] and r["delta3"] is not None and r["n_est"] >= 2:
        d3.append(r["delta3"])
tau3 = float(np.percentile(d3, 97))
dev_tri = [evaluate_event(loc, ev, k3_tau=tau3) for ev in dev["triple"]]
print("tau3", round(tau3, 1), "dev triples all3", summarize(dev_tri, 3)["all_sources_recovered"], round(time.time() - t), "s", flush=True)

array, te = make_stress("p10test", n_per_cell=20)
out = {"versions": versions(), "calibration_phase9": frozen_calibration().to_json(), "k3_tau": tau3,
       "k3_rule": "97th percentile of delta3 over p10dev resolvable pairs declared >=2 (no tuning on p10test)",
       "gate_um": 60.0, "pools": {"dev": "p10dev (K=3 threshold only)", "test": "p10test (reported, single run)"}}
rows = {fam: [evaluate_event(loc, ev) for ev in evs] for fam, evs in te.items()}
rows3 = {fam: [evaluate_event(loc, ev, k3_tau=tau3) for ev in te[fam]] for fam in ("triple", "single", "dt", "sep")}
print("base done", round(time.time() - t), "s", flush=True)

def bins(rs, key, edges, n_true=2, res_only=False):
    o = {}
    for lo, hi in edges:
        sel = [r for r in rs if lo <= r[key] <= hi and (r["res"] or not res_only)]
        o[f"{lo}-{hi}"] = summarize(sel, n_true)
    return o

out["dt_sweep_pairs"] = bins(rows["dt"], "dt", ((0, 0), (1, 3), (4, 9), (10, 20)))
out["dt_sweep_resolvable_pairs"] = bins(rows["dt"], "dt", ((0, 0), (1, 3), (4, 9), (10, 20)), res_only=True)
out["sep_sweep_pairs"] = bins(rows["sep"], "sep", ((0, 40), (40, 60), (60, 90), (90, 1000)))
out["dt_sweep_by_noise"] = {str(n): bins([r for r in rows["dt"] if r["noise"] == n], "dt", ((0, 0), (1, 3), (4, 9), (10, 20)))
                            for n in (5.0, 10.0, 20.0)}
out["singles_K2"] = summarize(rows["single"], 1)
out["triples_K2_pipeline"] = summarize(rows["triple"], 3)
out["triples_K2_pipeline_resolvable"] = summarize([r for r in rows["triple"] if r["res"]], 3)
out["triples_K3_extension"] = summarize(rows3["triple"], 3)
out["triples_K3_extension_resolvable"] = summarize([r for r in rows3["triple"] if r["res"]], 3)
out["singles_K3_extension"] = summarize(rows3["single"], 1)
pairs3 = [r for r in rows3["dt"] + rows3["sep"] if r["res"]]
out["resolvable_pairs_K3_extension"] = summarize(pairs3, 2)
out["resolvable_pairs_false_three_rate"] = float(np.mean([r["n_est"] == 3 for r in pairs3]))
out["resolvable_pairs_K2"] = summarize([r for r in rows["dt"] + rows["sep"] if r["res"]], 2)

# dropout during overlap: resolvable pairs + singles
rng = np.random.default_rng(20271003)
respairs = [ev for ev in te["dt"] + te["sep"] if ev.resolvable]
singles = te["single"]
drop = {}
for kind, frac in (("random", 0.75), ("random", 0.5), ("clustered", 0.75)):
    for mode in ("known", "unknown"):
        rp, rs = [], []
        for evs, acc in ((respairs, rp), (singles, rs)):
            for ev in evs:
                keep = make_mask(array.n_channels, frac, kind, rng)
                acc.append(evaluate_event(loc, ev, mask=keep) if mode == "known" else evaluate_event(loc, ev, zero=~keep))
        drop[f"{mode}_{kind}_{frac}"] = {"resolvable_pairs": summarize(rp, 2), "singles": summarize(rs, 1)}
out["dropout_during_overlap"] = drop
out["n"] = {k: len(v) for k, v in te.items()}
out["runtime_s"] = time.time() - t
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps({k: out[k] for k in ("resolvable_pairs_K2", "singles_K2", "triples_K2_pipeline", "triples_K3_extension",
                                      "singles_K3_extension", "resolvable_pairs_false_three_rate")}, indent=0, default=float)[:3000])
