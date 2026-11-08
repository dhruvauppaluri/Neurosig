"""Experiment X2 on development windows (A-D): P9 vs P10 (dead contacts excluded, engine rebuilt) with the Phase 9
battery (identical seeded masks/perturbations for both)."""
import json, sys
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, p9_factory
from edge.phase9.evalset import build_events_shift, evaluate
from edge.phase10.configs import P10_HASH, p10_factory
wins = H.manifest_windows("dev")
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
res = {"P9": evaluate(p9_factory, real, null), "P10": evaluate(p10_factory, real, null)}
json.dump({"p9_hash": P9_HASH, "p10_hash": P10_HASH, "results": res}, open("reports/phase10/exp_x2_dev.json", "w"), indent=1, default=float)
keys = ("dropU_random_0.75", "dropU_random_0.5", "dropU_clustered_0.75", "drop_random_0.5")
for m in res["P9"]:
    for k in ("P9", "P10"):
        x = res[k][m]
        print(f"{m:14s} {k:4s} valid {x['valid_fraction']:.2f} spread {x['within_unit_spread_um']['median']:5.1f} r/n {x['real_over_null_spread']:.2f} "
              + " ".join(f"{c}: {x[c]['median']:5.1f}/{x[c]['p90']:5.1f} fail {x[c]['failure_fraction']:.2f}" for c in keys))
