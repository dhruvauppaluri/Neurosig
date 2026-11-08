"""Reserve the Phase 11 FRESH LOCKED evaluation set (4 insertions never used in Phases 3-10) BEFORE any evaluation.

Insertions come from reports/phase11/insertion_search.json: >=32 VIS contacts, raw AP present, distinct labs, labs not
used before preferred, deterministic order. Only the probe sync file is read (recording length). Seeded; refuses to
overwrite. Run with data/ibl-venv.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from one.api import ONE

OUT = ROOT / "reports/phase11/split_manifest_phase11.json"
SEED = 20271107
if OUT.exists():
    raise SystemExit("manifest exists; the Phase 11 locked set must not be redrawn")
cands = [c for c in json.loads((ROOT / "reports/phase11/insertion_search.json").read_text())
         if c.get("vis", 0) >= 32 and c.get("raw_ap")]
cands = sorted(cands, key=lambda c: (not c["new_lab"], c["pid"]))
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
          silent=True, cache_dir="data/neuropixels/ibl_cache")
rng = np.random.default_rng(SEED)
sessions, wins, labs, skipped = {}, [], set(), []
for c in cands:
    if len(sessions) == 4:
        break
    if c["lab"] in labs:
        continue
    key = "MNOP"[len(sessions)]
    try:
        eid, probe = one.pid2eid(c["pid"])
        ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy", collection=f"raw_ephys_data/{probe}"), float)
    except Exception as e:  # noqa: BLE001
        skipped.append({"pid": c["pid"], "reason": f"{type(e).__name__}: {str(e)[:80]}"}); continue
    if ts.shape[0] < 3:      # identity/degenerate sync file: the probe-clock correction cannot be validated
        skipped.append({"pid": c["pid"], "lab": c["lab"], "reason": f"degenerate probe sync file shape {ts.shape}"}); continue
    total = float(ts[-1, 0] / 30000.0)
    sessions[key] = dict(pid=c["pid"], lab=c["lab"], probe_duration_s=total, vis_contacts=c["vis"], new_lab=c["new_lab"])
    labs.add(c["lab"])
    slots = np.arange(300.0, total - 180.0, 60.0)
    for i in rng.choice(len(slots), size=6, replace=False):
        wins.append(dict(session=key, pid=c["pid"], t0_s=round(float(slots[i] + rng.uniform(0, 30)), 3), duration_s=2.0,
                         split="locked_final_p11"))
man = dict(seed=SEED, window_s=2.0, skipped_for_availability=skipped,
           rule="Phase 11 final windows are reserved before any Phase 11 evaluation and evaluated ONCE after all Phase 11 configurations "
                "and analysis protocols are frozen; no parameter changes after viewing them",
           sessions=sessions, windows=wins)
OUT.write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(sessions, indent=1), len(wins), "windows")
