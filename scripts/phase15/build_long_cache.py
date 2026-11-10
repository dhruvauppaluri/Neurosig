"""Detector-event caches for the 30 s contiguous windows (data/phase15/det_long_<s>.npz). Usage: build_long_cache.py A,B,C,D,L,O|final"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase14 import detcache as DC
from edge.phase12.splits import split_windows
if sys.argv[1] == "final":
    S = json.loads(Path("reports/phase15/split_manifest_phase15.json").read_text())["sessions"]
    items = [(k, dict(session=k, pid=v["pid"], t0_s=v["long_t0_s"], duration_s=30.0), Path(f"data/phase15/det_long_{k}.npz")) for k, v in S.items()]
else:
    by = defaultdict(list)
    for sp in ("dev", "select"):
        for w in split_windows(sp):
            by[w["session"]].append(w)
    items = []
    for s in sys.argv[1].split(","):
        w0 = min(by[s], key=lambda w: w["t0_s"])
        items.append((s, dict(session=s, pid=w0["pid"], t0_s=w0["t0_s"], duration_s=30.0), Path(f"data/phase15/det_long_{s}.npz")))
for s, w, p in items:
    if not p.exists():
        print(s, DC.build(s, [w], p, log=lambda m: None), flush=True)
