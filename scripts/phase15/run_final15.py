"""Phase 15 single-use FINAL evaluation of the frozen configurations (reports/phase15/frozen_p15.json).

Fresh data reserved before Phase 15 optimization: IBL insertions X1-X4 (reports/phase15/split_manifest_phase15.json;
12 x 2 s windows + one contiguous 30 s window each) and 6 never-opened Allen mice (reports/phase15/allen_split15.json).
No fresh MEArec morphology exists (all 13 BBP L5 models were used by Phase 14), so the biophysical count / position
results of Phase 15 are cross-validated (leave-one-morphology-out), never called final.
Steps (refuses to run twice):
  1. caches for X1-X4 (detector events for 2 s windows and the 30 s window; sorter-aligned raw segments)
  2. identity: OU-P14 vs ID-P15 on the 2 s windows and the 30 s windows
  3. real-data count proxy: P14-C (Phase 14) on isolated vs coincident sorter events and pooled templates
  4. IBL visual decoding with the frozen Phase 13 protocol on X1-X4
  5. Allen cross-animal decoding of the frozen D-P15 spec, the Phase 14 reference spec and the e11 spec (needed only
     for the pre-registered secondary per-task routing) on the new mice
The streaming benchmark (scripts/phase15/stream15.py) measures runtime, not accuracy, and runs on development sessions.
"""
import json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("reports/phase15/final15.json")
if OUT.exists():
    raise SystemExit("single use")
fr = json.loads(Path("reports/phase15/frozen_p15.json").read_text())
log = open("data/phase15/final15.log", "a")


def run(*a, py=".venv/bin/python", env=None):
    import os
    t = time.time(); print("RUN", *a, flush=True)
    e = os.environ.copy(); e.update(env or {}); e["PYTHONPATH"] = "."
    r = subprocess.run([py, *a], stdout=log, stderr=subprocess.STDOUT, env=e)
    print("  rc", r.returncode, f"{time.time() - t:.0f}s", flush=True)
    return r.returncode


from edge.phase12 import pooling as PL
from edge.phase14 import detcache as DC
man = json.loads(Path("reports/phase15/split_manifest_phase15.json").read_text())
by = defaultdict(list)
for w in man["windows"]:
    by[w["session"]].append(w)
for s, ws in sorted(by.items()):
    ws = sorted(ws, key=lambda w: w["t0_s"])
    if not Path(f"data/phase15/det_{s}.npz").exists():
        DC.build(s, ws, Path(f"data/phase15/det_{s}.npz"), log=lambda m: None)
    if not Path(f"data/phase15/sess_{s}.npz").exists():
        se = PL.collect_session(ws, s)
        if se is not None:
            PL.save_session(se, Path(f"data/phase15/sess_{s}.npz"))
idp = json.dumps(fr["identity"]["ID-P15"])
steps = {
    "long_cache": run("scripts/phase15/build_long_cache.py", "final"),
    "id_oup14": run("scripts/phase15/run_identity.py", "final", "{}", "final_oup14"),
    "id_p15": run("scripts/phase15/run_identity.py", "final", idp, "final_idp15"),
    "id_oup14_long": run("scripts/phase15/run_identity.py", "final_long", "{}", "final_long_oup14"),
    "id_p15_long": run("scripts/phase15/run_identity.py", "final_long", idp, "final_long_idp15"),
    "real_count": run("scripts/phase15/real_count15.py", "final"),
    "ibl_decode": run("scripts/phase15/run_ibl15.py", py="data/ibl-venv/bin/python"),
    "allen_dp15": run("scripts/phase15/run_xanimal.py", "final", json.dumps(fr["vision"]["D-P15"]), "final_dp15", py="data/ibl-venv/bin/python",
                      env={"OMP_NUM_THREADS": "4"}),
    "allen_ref14": run("scripts/phase15/run_xanimal.py", "final", json.dumps(fr["vision"]["reference_p14"]), "final_ref14", py="data/ibl-venv/bin/python",
                       env={"OMP_NUM_THREADS": "4"}),
    "allen_e11": run("scripts/phase15/run_xanimal.py", "final", json.dumps(fr["vision"]["e11_layer_mlp"]), "final_e11", py="data/ibl-venv/bin/python",
                     env={"OMP_NUM_THREADS": "4"}),
}
OUT.write_text(json.dumps({"frozen_hash": fr["hash"], "steps_returncode": steps}, indent=1) + "\n")
print(steps)
