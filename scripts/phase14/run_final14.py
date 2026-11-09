"""Phase 14 single-use FINAL evaluation of the frozen P14 configuration (reports/phase14/frozen_p14.json).

Runs once (refuses if reports/phase14/final14.json exists):
  1. biophysical count / position / uncertainty on unseen morphologies: p14final (NP1) and p14final_np2 (NP2), P14
     pipeline + P12-LR baseline on identical events
  2. drift streams on unseen morphologies (identity vs true labels, Kalman tracker)
  3. fresh IBL sessions T-W: detector-event caches, online association (frozen OU-P14), unit-level spread,
     real-recording count check (isolated / coincident / pooled templates)
Usage: .venv/bin/python scripts/phase14/run_final14.py
"""
import json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("reports/phase14/final14.json")
if OUT.exists():
    raise SystemExit("final14.json exists: the Phase 14 final evaluation is single-use")
fr = json.loads(Path("reports/phase14/frozen_p14.json").read_text())
py = ".venv/bin/python"
log = open("data/phase14/final14.log", "a")


def run(*a):
    t = time.time(); print("RUN", *a, flush=True)
    r = subprocess.run([py, *a], stdout=log, stderr=subprocess.STDOUT)
    print("  rc", r.returncode, f"{time.time() - t:.0f}s", flush=True)
    return r.returncode


# 3a. real-data caches for T-W (sorter used only for evaluation fields)
from edge.phase12 import pooling as PL
from edge.phase14 import detcache as DC
man = json.loads(Path("reports/phase14/split_manifest_phase14.json").read_text())
by = defaultdict(list)
for w in man["windows"]:
    by[w["session"]].append(w)
inv = {}
for s, ws in sorted(by.items()):
    ws = sorted(ws, key=lambda w: w["t0_s"])
    p = Path(f"data/phase14/det_{s}.npz")
    if not p.exists():
        inv[s] = DC.build(s, ws, p, log=lambda m: None)
    q = Path(f"data/phase14/sess_{s}.npz")
    if not q.exists():
        se = PL.collect_session(ws, s)
        if se is not None:
            PL.save_session(se, q)
print("caches", inv, flush=True)
steps = {
    "count_np1": run("scripts/phase14/run_p14.py", "p14final"),
    "count_np2": run("scripts/phase14/run_p14.py", "p14final_np2"),
    "eval_np1": run("scripts/phase14/eval_p14.py", "evaluate", "p14final", ""),
    "eval_np2": run("scripts/phase14/eval_p14.py", "evaluate", "p14final_np2", ""),
    "baseline_np1": run("scripts/phase14/run_14_2.py", "p14final", "1", "4"),
    "baseline_np2": run("scripts/phase14/run_14_2.py", "p14final_np2", "1", "4"),
    "drift": run("scripts/phase14/run_14_5.py", "p14final", "3"),
    "online": run("scripts/phase14/run_14_4.py", "final", "min_gain=80", "merge_every=1000000000", "snr_assign=6", "--tag=oup14"),
    "real_count": run("scripts/phase14/run_14_3_real.py", "final"),
}
OUT.write_text(json.dumps({"frozen_hash": fr["hash"], "steps_returncode": steps, "det_caches": inv,
                           "rule": "single use; numbers in the referenced per-step reports"}, indent=1) + "\n")
print(steps)
