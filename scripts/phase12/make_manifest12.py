"""Reserve the Phase 12 FRESH final evaluation set (sessions Q-T: 4 insertions never used in Phases 3-11) BEFORE any
Phase 12 evaluation. Same eligibility rules as Phase 11 (>= 32 VIS contacts, raw AP present, non-degenerate probe sync
file, distinct labs; labs never used before preferred, deterministic order), 8 windows of 2 s per insertion so that
unit-level pooling has enough spikes per unit. Seeded; refuses to overwrite. Run with data/ibl-venv."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from one.api import ONE

OUT = ROOT / "reports/phase12/split_manifest_phase12.json"
SEED = 20271301
if OUT.exists():
    raise SystemExit("manifest exists; the Phase 12 final set must not be redrawn")
used_pids, used_labs = set(), set()
for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json",
          "reports/phase10/split_manifest_phase10.json", "reports/phase11/split_manifest_phase11.json"):
    for v in json.loads((ROOT / f).read_text())["sessions"].values():
        used_pids.add(v["pid"]); used_labs.add(v.get("lab"))
cands = [c for c in json.loads((ROOT / "reports/phase11/insertion_search.json").read_text())
         if c.get("vis", 0) >= 32 and c.get("raw_ap") and c["pid"] not in used_pids]
cands = sorted(cands, key=lambda c: (c["lab"] in used_labs, c["pid"]))
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
          silent=True, cache_dir="data/neuropixels/ibl_cache")
rng = np.random.default_rng(SEED)
sessions, wins, labs, skipped = {}, [], set(), []
for c in cands:
    if len(sessions) == 4:
        break
    if c["lab"] in labs:
        continue
    key = "QRST"[len(sessions)]
    try:
        eid, probe = one.pid2eid(c["pid"])
        ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy", collection=f"raw_ephys_data/{probe}"), float)
    except Exception as e:  # noqa: BLE001
        skipped.append({"pid": c["pid"], "reason": f"{type(e).__name__}: {str(e)[:80]}"}); continue
    if ts.shape[0] < 3:
        skipped.append({"pid": c["pid"], "lab": c["lab"], "reason": f"degenerate probe sync file shape {ts.shape}"}); continue
    total = float(ts[-1, 0] / 30000.0)
    sessions[key] = dict(pid=c["pid"], lab=c["lab"], probe_duration_s=total, vis_contacts=c["vis"],
                         lab_new_in_phase12=c["lab"] not in used_labs)
    labs.add(c["lab"])
    slots = np.arange(300.0, total - 180.0, 60.0)
    for i in rng.choice(len(slots), size=8, replace=False):
        wins.append(dict(session=key, pid=c["pid"], t0_s=round(float(slots[i] + rng.uniform(0, 30)), 3), duration_s=2.0,
                         split="final_p12"))
man = dict(seed=SEED, window_s=2.0, skipped_for_availability=skipped,
           rule="Phase 12 final windows are reserved before any Phase 12 evaluation and evaluated ONCE by "
                "scripts/phase12/run_final12.py after all Phase 12 configurations and analysis protocols are frozen",
           sessions=sessions, windows=wins)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(sessions, indent=1), len(wins), "windows", "skipped", skipped)
