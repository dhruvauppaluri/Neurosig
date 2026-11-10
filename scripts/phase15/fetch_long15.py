"""Phase 15: contiguous 30 s raw AP windows for online identity / drift work (data/ibl-venv). One window per session,
starting at the session's first previously cached window t0 (dev/select sessions; data already used in earlier
phases) or, for the Phase 15 final sessions, at a seeded time after 300 s (reports/phase15/split_manifest_phase15.json).
Usage: fetch_long15.py A,B,C,D,L,O | fetch_long15.py final"""
import json, socket, sys, time
from collections import defaultdict
from pathlib import Path
socket.setdefaulttimeout(300)
sys.path.insert(0, ".")
from scripts.phase5.data import fetch_window
from edge.phase12.splits import split_windows
DUR = 30.0
if sys.argv[1] == "final":
    wins = [dict(session=k, pid=v["pid"], t0_s=v["long_t0_s"]) for k, v in json.loads(Path("reports/phase15/split_manifest_phase15.json").read_text())["sessions"].items()]
else:
    by = defaultdict(list)
    for sp in ("dev", "select"):
        for w in split_windows(sp):
            by[w["session"]].append(w)
    wins = [dict(session=s, pid=min(by[s], key=lambda w: w["t0_s"])["pid"], t0_s=min(w["t0_s"] for w in by[s])) for s in sys.argv[1].split(",")]
for w in wins:
    for attempt in range(3):
        t = time.time()
        try:
            p = fetch_window(w["pid"], w["t0_s"], DUR); print("ok", w["session"], w["t0_s"], p, f"{time.time()-t:.0f}s", flush=True); break
        except Exception as e:  # noqa: BLE001
            print("retry", w["session"], type(e).__name__, str(e)[:100], flush=True); time.sleep(10)
