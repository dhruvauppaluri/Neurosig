"""Phase 14.2-14.3 rule comparison on dev / select (and NP2 select geometry); writes reports/phase14/exp14_3_<tag>.json.

Thresholds:
  dev-calibrated   quantile of p14dev singles (fixed number reused everywhere: the Phase 12 protocol)
  self-calibrated  quantile of the pool's label-free calibration singles (development morphologies in the pool's own
                   background noise, bench14.generate(calibration=True)); the deployable variant
  oracle matched   quantile of the pool's own singles: matched-false-positive comparison only, never a deployable rule
Usage: .venv/bin/python scripts/phase14/select_14_3.py [tag]
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase14 import count_eval as CE

TAG = sys.argv[1] if len(sys.argv) > 1 else "select"
RULES = ["lr", "line20_F", "ms_F", "ms5_F", "ms_st_F", "ms5_st_F", "line20_sp_F", "ms_sp_F", "ms5_sp_F", "msq_sp_F",
         "msn_sp_F", "msw_sp_F", "ms3_sp_F", "ms_Ftr", "learn", "learn_sp"]
L = lambda p: pickle.load(open(p, "rb")) if Path(p).exists() else None
dev, sel, np2 = L("data/phase14/rows_p14dev.pkl"), L("data/phase14/rows_p14select.pkl"), L("data/phase14/rows_p14sel_np2.pkl")
cal = {"p14select": L("data/phase14/rows_cal_p14select.pkl"), "p14sel_np2": L("data/phase14/rows_cal_p14sel_np2.pkl"),
       "p14dev": dev}
models = {"learn": CE.train_learned(dev), "learn_sp": CE.train_learned(dev, suf="_sp")}
out = {"versions": versions(), "rules": {}}
for rule in RULES:
    m = models.get(rule)
    try:
        thr_dev = CE.calibrate(dev, rule, 0.02, m)
    except KeyError:
        continue
    o = {}
    for nm, rows in (("p14dev", dev), ("p14select", sel), ("p14sel_np2", np2)):
        if rows is None:
            continue
        o[nm] = {"dev_cal": CE.evaluate(rows, rule, thr_dev, m, thr3=None)}
        c = cal.get(nm)
        if c is not None and rule != "lr":
            o[nm]["self_cal"] = CE.evaluate(rows, rule, CE.calibrate(c, rule, 0.02, m), m)
        o[nm]["oracle_2pct"] = CE.evaluate(rows, rule, CE.calibrate(rows, rule, 0.02, m), m)
        o[nm]["oracle_5pct"] = CE.evaluate(rows, rule, CE.calibrate(rows, rule, 0.05, m), m)
        o[nm]["pr_curve"] = CE.pr_curve(rows, rule, m)
    out["rules"][rule] = o
    line = [rule]
    for nm in o:
        for k in ("dev_cal", "self_cal", "oracle_2pct"):
            if k in o[nm]:
                e = o[nm][k]
                line.append(f"{nm[3:]}:{k[:4]} FT {e['false_two']['rate']:.3f} R {e['pairs_resolvable']['recall']:.2f}")
        line.append(f"AUC {o[nm]['dev_cal']['auc_single_vs_res']:.3f}")
    print(" | ".join(line), flush=True)
Path(f"reports/phase14/exp14_3_{TAG}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
