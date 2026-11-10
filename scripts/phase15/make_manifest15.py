"""Reserve the Phase 14 FRESH final evaluation sessions (T-W) BEFORE any Phase 14 evaluation.

Candidates: reports/phase14/insertion_search14.json (>= 32 VIS contacts, raw AP), never used in Phases 3-13, minus the
two insertions with degenerate probe-sync files skipped in Phase 10 (0393f34c, 071f02e7). Distinct labs, deterministic
order (pid), non-degenerate _spikeglx_*.timestamps.npy. 12 windows of 2 s per insertion (seed 20271421), spread over
the recording after t = 300 s, so that unit pooling, online grouping and drift tracking have spikes from across the
session. The same insertions are the fresh final sessions for IBL visual decoding (Phase 14V).
Seeded; refuses to overwrite. Run with data/ibl-venv.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from one.api import ONE

OUT = ROOT / "reports/phase15/split_manifest_phase15.json"
SEED = 20271521
DEGENERATE = {"0393f34c-a2bd-4c01-99c9-f6b4ec6e786d", "071f02e7-752a-4094-af79-8dd764e9d85d"}
if OUT.exists():
    raise SystemExit("manifest exists; the Phase 15 final set must not be redrawn")
used = {v["pid"] for v in json.loads((ROOT / "reports/phase14/split_manifest_phase14.json").read_text())["sessions"].values()}
DEGENERATE = DEGENERATE | {"176b4fe3-f570-4d9f-9e25-a5d218f75c8b"}       # Phase 14 skip (degenerate sync)
cands = [c for f in ("reports/phase14/insertion_search14.json", "reports/phase15/insertion_search15.json")
         for c in json.loads((ROOT / f).read_text())
         if c.get("vis", 0) >= 32 and c.get("raw_ap") and c["pid"] not in DEGENERATE and c["pid"] not in used]
cands = sorted(cands, key=lambda c: c["pid"])
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
          silent=True, cache_dir="data/neuropixels/ibl_cache")
rng = np.random.default_rng(SEED)
sessions, wins, labs, skipped = {}, [], set(), []
for c in cands:
    if len(sessions) == 4:
        break
    if c["lab"] in labs:
        continue
    key = ["X1", "X2", "X3", "X4"][len(sessions)]
    try:
        eid, probe = one.pid2eid(c["pid"])
        ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy", collection=f"raw_ephys_data/{probe}"), float)
    except Exception as e:  # noqa: BLE001
        skipped.append({"pid": c["pid"], "reason": f"{type(e).__name__}: {str(e)[:80]}"}); continue
    if ts.shape[0] < 3:
        skipped.append({"pid": c["pid"], "lab": c["lab"], "reason": f"degenerate probe sync file shape {ts.shape}"}); continue
    total = float(ts[-1, 0] / 30000.0)
    sessions[key] = dict(pid=c["pid"], eid=str(eid), lab=c["lab"], probe_duration_s=total, vis_contacts=c["vis"],
                         long_t0_s=round(float(rng.uniform(300.0, total - 240.0)), 3))
    labs.add(c["lab"])
    slots = np.arange(300.0, total - 180.0, 60.0)
    for i in np.sort(rng.choice(len(slots), size=12, replace=False)):
        wins.append(dict(session=key, pid=c["pid"], t0_s=round(float(slots[i] + rng.uniform(0, 30)), 3), duration_s=2.0,
                         split="final_p15"))
man = dict(seed=SEED, window_s=2.0, skipped_for_availability=skipped,
           rule="Phase 15 final sessions are reserved before any Phase 14 evaluation and evaluated ONCE by "
                "scripts/phase15/run_final15.py (localization) and scripts/phase14/run_vis14.py final (decoding) after "
                "all Phase 14 configurations are frozen in reports/phase15/frozen_p15.json",
           sessions=sessions, windows=wins)
OUT.write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(sessions, indent=1), len(wins), "windows", "skipped", skipped)
