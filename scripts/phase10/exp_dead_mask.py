"""Experiment X1: unknown dead contacts -> fit mask instead of interpolation, for the K<=2 overlap pipeline.
Decision on p10dev; confirmation (both arms, run once) on the fresh p10confirm pool with --confirm."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9.harness import make_mask
from edge.phase10.overlap_stress import evaluate_event, localizer, make_stress, summarize

pool = "p10confirm" if "--confirm" in sys.argv else "p10dev"
out_path = Path(f"reports/phase10/exp_dead_mask_{pool}.json")
if pool == "p10confirm" and out_path.exists():
    raise SystemExit("confirmation pool already used")
array, ev = make_stress(pool, n_per_cell=12 if pool == "p10dev" else 20)
loc = localizer(array)
pairs = [e for e in ev["dt"] + ev["sep"] if e.resolvable]; singles = ev["single"]
res = {"pool": pool, "n_resolvable_pairs": len(pairs), "n_singles": len(singles), "conditions": {}}
for kind, frac in (("none", 1.0), ("random", 0.75), ("random", 0.5), ("clustered", 0.75)):
    for arm in ("interp_P9", "mask_X1"):
        rng = np.random.default_rng(20271005)          # identical masks for both arms
        rp, rs = [], []
        for evs, acc in ((pairs, rp), (singles, rs)):
            for e in evs:
                keep = np.ones(array.n_channels, bool) if kind == "none" else make_mask(array.n_channels, frac, kind, rng)
                acc.append(evaluate_event(loc, e, zero=None if kind == "none" else ~keep, dead_as_mask=(arm == "mask_X1")))
        sp, ss = summarize(rp, 2), summarize(rs, 1)
        res["conditions"][f"{kind}_{frac}/{arm}"] = {"resolvable_pairs": sp, "singles": ss,
                                                     "false_two_singles": float(np.mean([r["n_est"] >= 2 for r in rs]))}
        print(f"{kind}_{frac:<5} {arm:10s} pairs recall {sp['all_sources_recovered']:.3f} err {sp['matched_err_median_um']:.1f} "
              f"acc {sp['accepted']:.2f} | singles false2 {res['conditions'][f'{kind}_{frac}/{arm}']['false_two_singles']:.3f} "
              f"err {ss['matched_err_median_um']:.1f} acc {ss['accepted']:.2f}", flush=True)
out_path.write_text(json.dumps(res, indent=1, default=float) + "\n")
