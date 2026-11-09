"""Build Phase 14 detector-event caches for the development / selection sessions (A-S; all previously used).
The fresh final sessions T-W are cached only by run_final14.py, after the P14 configuration is frozen.
Usage: .venv/bin/python scripts/phase14/build_det_cache.py [sessions]"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase12.splits import split_windows
from edge.phase14 import detcache as DC

only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
by = defaultdict(list)
for split in ("dev", "select", "final"):
    for w in split_windows(split):
        by[w["session"]].append(w)
inv_p = Path("reports/phase14/det_cache_inventory.json")
inv = json.loads(inv_p.read_text()) if inv_p.exists() else {}
for s, ws in sorted(by.items()):
    if only and s not in only:
        continue
    out = Path(f"data/phase14/det_{s}.npz")
    if out.exists():
        continue
    t = time.time()
    inv[s] = {**DC.build(s, sorted(ws, key=lambda w: w["t0_s"]), out, log=lambda m: None), "n_windows": len(ws),
              "seconds": round(time.time() - t)}
    print(s, inv[s], flush=True)
    inv_p.write_text(json.dumps(inv, indent=1) + "\n")
