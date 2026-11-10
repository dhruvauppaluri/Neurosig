"""Phase 15 Allen data (data/allen-venv). Reserves the Phase 15 split of the 15 DANDI 000021 sessions never opened in
Phases 13-14 BEFORE opening any of them, then streams them with the corrected Phase 14 cache format (5 ms bins,
local_index waveform map) plus receptive-field Gabor positions.

Split (seed 20271501, reports/phase15/allen_split15.json): seeded permutation of the unused sessions; the first 8 are
FINAL candidates (first 6 eligible used), the rest train expansion. All Phase 13-14 mice (17, including the Phase 14
final mice, now a used benchmark) are development / selection data in Phase 15.
Usage: fetch_allen15.py split | fetch_allen15.py fetch
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPLIT = ROOT / "reports/phase15/allen_split15.json"; INV = ROOT / "reports/phase15/allen_inventory15.json"


def make_split():
    if SPLIT.exists():
        raise SystemExit("split exists")
    s14 = json.loads((ROOT / "reports/phase14/allen_split14.json").read_text())
    inv14 = json.loads((ROOT / "reports/phase14/allen_inventory14.json").read_text())
    unused = sorted(x for x in s14["final_candidates"] + s14["train_candidates"] if x not in inv14)
    perm = np.random.default_rng(20271501).permutation(len(unused))
    used = sorted(s for s, r in inv14.items() if r.get("status") == "ok")
    out = {"seed": 20271501, "rule": "reserved before opening any listed session; final = first 6 eligible of final_candidates",
           "phase14_used_ok": used, "final_candidates": [unused[i] for i in perm[:8]], "train_candidates": [unused[i] for i in perm[8:]]}
    SPLIT.write_text(json.dumps(out, indent=1) + "\n"); print(out)


def fetch():
    import socket
    socket.setdefaulttimeout(300)              # a read that stalls for 5 min raises instead of hanging forever
    import scripts.phase14.fetch_allen14 as F
    s = json.loads(SPLIT.read_text())
    inv = json.loads(INV.read_text()) if INV.exists() else {}
    by = {F.sess_of(a): a for a in F.ASSETS}
    n_final = 0
    for grp in ("train_candidates", "final_candidates"):
        for sess in s[grp]:
            if grp == "final_candidates" and n_final >= 6:
                break
            if sess in inv and inv[sess]["status"] == "ok":
                n_final += grp == "final_candidates"; continue
            t = time.time()
            for attempt in range(3):           # network stalls (socket timeout) are retried; data errors are not
                try:
                    r = F.process(by[sess]); break
                except (TimeoutError, OSError) as e:
                    r = {"status": f"failed: {type(e).__name__}: {str(e)[:100]}"}
                    print(grp, sess, "network error, retry", attempt + 1, flush=True)
                except Exception as e:  # noqa: BLE001
                    r = {"status": f"failed: {type(e).__name__}: {str(e)[:100]}"}; break
            r["group"] = grp; inv[sess] = r
            n_final += (grp == "final_candidates") and r["status"] == "ok"
            print(grp, sess, r["status"], r.get("n_vis_good_units"), f"{time.time() - t:.0f}s", flush=True)
            INV.write_text(json.dumps(inv, indent=1, default=str) + "\n")


if __name__ == "__main__":
    make_split() if sys.argv[1] == "split" else fetch()
