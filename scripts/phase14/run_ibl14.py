"""Phase 14V IBL replication: the frozen Phase 13 IBL protocol (scripts/phase13/run_ibl13.py, unchanged) applied ONCE
to the fresh Phase 14 final insertions T-W (never decoded before). Tests whether the Phase 13 temporal advantage on
contralateral contrast level (+3.0 select, +5.5 final) replicates on new animals and labs, together with every
Phase 13 control (shuffled labels, pre-stimulus window, still trials, choice decoding, error trials).
Run with data/ibl-venv: PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/run_ibl14.py
Writes reports/phase14/ibl14_final.json (refuses to overwrite)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase13 import decode as D
import scripts.phase13.run_ibl13 as R

OUT = Path("reports/phase14/ibl14_final.json")
if OUT.exists():
    raise SystemExit("single use")
if not Path("reports/phase14/frozen_p14.json").exists():
    raise SystemExit("frozen_p14.json required")
t0 = time.time(); res = {}
for key in "TUVW":
    try:
        res[key] = R.run_session(key)
    except Exception as e:  # noqa: BLE001
        res[key] = {"status": f"failed: {type(e).__name__}: {str(e)[:100]}"}
    print(key, res[key].get("status", "ok"), flush=True)
summary = {}
for t in ("T1_presence", "T2_side", "T3_contrast5", "T5_choice"):
    S = [r["tasks"][t] for r in res.values() if "tasks" in r and t in r["tasks"]]
    if not S:
        continue
    keys = set.intersection(*[set(s) for s in S])
    summary[t] = {"n_sessions": len(S),
                  "balanced_accuracy_mean": {k: float(np.mean([s[k]["balanced_accuracy"] for s in S])) for k in keys if isinstance(S[0][k], dict) and "balanced_accuracy" in S[0][k]},
                  "majority_mean": float(np.mean([s["lr:counts"]["majority"] for s in S]))}
    pk = set.intersection(*[set(s.get("paired", {})) for s in S]) if all("paired" in s for s in S) else set()
    summary[t]["paired_across_sessions"] = {k: D.across_sessions([s["paired"][k]["delta"] for s in S]) for k in pk}
OUT.write_text(json.dumps({"protocol": "Phase 13 run_ibl13.py (frozen), sessions T-W", "sessions": res, "summary": summary,
                           "runtime_s": time.time() - t0}, indent=1, default=float) + "\n")
print(json.dumps({t: {k: v for k, v in s.items() if k != "paired_across_sessions"} for t, s in summary.items()}, indent=1)[:3000])
