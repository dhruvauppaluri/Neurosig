"""Reserve the Phase 6 LOCKED FINAL evaluation set (4 new insertions, never used before) BEFORE any evaluation.

Run with data/ibl-venv (reads only the small probe sync file for recording length). Seeded; refuses to overwrite.
The Phase 5 held-out windows are recorded as 'previously_used_benchmark', not as a fresh holdout.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from one.api import ONE

OUT = Path(__file__).resolve().parents[2] / "reports" / "phase6" / "split_manifest_phase6.json"
SEED = 20261106
NEW = {"E": ("c5b9e063-f640-4936-b851-f7602cb6659b", "hausserlab"), "F": ("5135e93f-2f1f-4301-9532-b5ad62548c49", "angelakilab"),
       "G": ("18d316bf-d322-4c5c-814e-a58147f7bf5f", "steinmetzlab"), "H": ("730770d6-617a-4ada-95db-a48521befda5", "churchlandlab_ucla")}
if OUT.exists():
    raise SystemExit("manifest exists; the locked set must not be redrawn")
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True, cache_dir="data/neuropixels/ibl_cache")
rng = np.random.default_rng(SEED)
wins, sessions = [], {}
for key, (pid, lab) in NEW.items():
    eid, probe = one.pid2eid(pid)
    ts = np.asarray(one.load_dataset(eid, "_spikeglx_*.timestamps.npy", collection=f"raw_ephys_data/{probe}"), dtype=float)
    total_s = float(ts[-1, 0] / 30000.0)
    sessions[key] = dict(pid=pid, lab=lab, probe_duration_s=total_s)
    slots = np.arange(300.0, total_s - 180.0, 60.0)
    pick = rng.choice(len(slots), size=6, replace=False)
    for i in pick:
        wins.append(dict(session=key, pid=pid, t0_s=round(float(slots[i] + rng.uniform(0, 30)), 3), duration_s=2.0, split="locked_final"))
prev = json.loads((OUT.parent.parent / "phase5" / "split_manifest.json").read_text())
man = dict(revision_note="Session E was first drawn as mainenlab 366c7c11 but that insertion has no raw AP files on the server (no .ap.ch); replaced by hausserlab c5b9e063 before any window was downloaded or evaluated (availability only).", seed=SEED, window_s=2.0, rule="locked_final windows are reserved before any evaluation, evaluated ONCE per frozen configuration; no parameter changes after viewing them",
           sessions=sessions, windows=wins,
           previously_used_benchmark="Phase 5 held-out windows (reports/phase5/split_manifest.json split=heldout) were evaluated and discussed; they are not a fresh holdout")
OUT.write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps({k: (round(v['probe_duration_s']), v['lab']) for k, v in sessions.items()}), len(wins), "windows")
