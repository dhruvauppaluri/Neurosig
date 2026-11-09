"""P12-LR and P11-style baselines on the identical Phase 14 FINAL events (rows_p14final*.pkl from run_14_2.py), next
to the frozen P14 rule computed in the same harness (SourceModel ms3 sp, c4r2 search) at its frozen dev-calibrated
operating point and at matched false-two rates. Writes reports/phase14/final14_baseline.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase14 import count_eval as CE
dev = pickle.load(open("data/phase14/rows_p14dev.pkl", "rb"))
out = {}
for pool in ("p14final", "p14final_np2"):
    rows = pickle.load(open(f"data/phase14/rows_{pool}.pkl", "rb"))
    o = {"P12-LR frozen threshold": CE.evaluate(rows, "lr", rows[0]["lr_thr"] if "lr_thr" in rows[0] else 0.5)}
    for fp in (0.02, 0.05):
        for rule in ("lr", "ms3_sp_F", "line20_F"):
            o[f"{rule} matched {fp}"] = CE.evaluate(rows, rule, CE.calibrate(rows, rule, fp))
    o["auc"] = {rule: CE.evaluate(rows, rule, 0.0)["auc_single_vs_res"] for rule in ("lr", "ms3_sp_F", "line20_F", "ms_F")}
    o["pr_curve"] = {rule: CE.pr_curve(rows, rule) for rule in ("lr", "ms3_sp_F")}
    out[pool] = o
    for k, e in o.items():
        if isinstance(e, dict) and "false_two" in e:
            print(pool, k, "FT", round(e["false_two"]["rate"], 3), "res", round(e["pairs_resolvable"]["recall"], 3), "sim", round(e["pairs_simultaneous_res"]["recall"], 3))
    print(pool, "AUC", {k: round(v, 3) for k, v in o["auc"].items()})
Path("reports/phase14/final14_baseline.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
