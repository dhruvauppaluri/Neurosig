"""Phase 11D: collect dropout records on dev (policy selection) and selection windows (confirmation), choose the
fallback policy on dev by a pre-registered rule, evaluate it on selection.
Rule: among policies (and thresholds) that keep >= 80% of P10's median improvement over P9 on dev and accept
>= 95% of events, pick the lowest dev p90 displacement; ties -> simplest. Applied per engine pooled over the three
unknown-dropout conditions."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.configs import p9_factory
from edge.phase9.evalset import build_events_shift
from edge.phase10.configs import p10_factory
from edge.phase11.dropout import POLICIES, apply, collect

def get(split):
    p = Path(f"data/phase11/dropout_{split}.pkl")
    if p.exists():
        return pickle.load(open(p, "rb"))
    wins = H.manifest_windows("dev") if split == "dev" else H.manifest_windows("heldout5") + H.manifest_windows("locked") + H.manifest_windows("locked10")
    rows = collect(build_events_shift(wins, 0.0), p9_factory, p10_factory)
    pickle.dump(rows, open(p, "wb")); return rows

dev, sel = get("dev"), get("select")
grid = {"P10": [0], "P9": [0], "dead_near>=k->interp": [1, 2, 3, 4, 6], "live_near<k->interp": [2, 3, 4, 6],
        "extrap->interp": [0], "disagree>k->reject": [20, 30, 50, 80], "disagree>k->interp": [10, 20, 30, 50]}
out = {"rule": __doc__, "dev": {}, "select": {}, "chosen": None}
cand = []
for pol, ks in grid.items():
    for k in ks:
        d = apply(dev, pol, k); s = apply(sel, pol, k)
        out["dev"][f"{pol}|{k}"] = d; out["select"][f"{pol}|{k}"] = s
p9d, p10d = out["dev"]["P9|0"], out["dev"]["P10|0"]
gain = p9d["median"] - p10d["median"]
for key, d in out["dev"].items():
    if d["accepted"] >= 0.95 and (p9d["median"] - d["median"]) >= 0.8 * gain:
        cand.append((d["p90"], len(key), key))
cand.sort()
out["chosen"] = cand[0][2]
for key in ("P9|0", "P10|0", out["chosen"]):
    print(f"{key:28s} dev {out['dev'][key]} | select {out['select'][key]}")
# per-condition breakdown on selection for P9 / P10 / chosen
pol, k = out["chosen"].split("|")
out["select_by_condition"] = {c: {n: apply([r for r in sel if r["cond"] == c], n.split("|")[0], int(n.split("|")[1]))
                                  for n in ("P9|0", "P10|0", out["chosen"])}
                              for c in sorted({r["cond"] for r in sel})}
out["select_by_engine"] = {e: {n: apply([r for r in sel if r["engine"] == e], n.split("|")[0], int(n.split("|")[1]))
                               for n in ("P9|0", "P10|0", out["chosen"])}
                           for e in sorted({r["engine"] for r in sel})}
Path("reports/phase11/exp11d_dropout.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["select_by_condition"], indent=0)[:2500])
