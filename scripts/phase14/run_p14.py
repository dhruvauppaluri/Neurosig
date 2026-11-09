"""Run the frozen-candidate P14 event pipeline on a Phase 14 bench pool and its label-free calibration set.

Usage: .venv/bin/python scripts/phase14/run_p14.py <pool> [n_cand rounds]
  writes data/phase14/p14rows_<pool>[_cXrY].pkl with per-event outputs (count statistic, positions, region deltas,
  runtimes) next to the truth. Thresholds and region levels are NOT applied here (they are calibrated on dev only
  by calib_p14.py and evaluated by eval_p14.py), except thr2 (self-calibrated per pool from its own calibration set,
  the deployable rule) which is needed to decide whether the K=3 stage runs.
Final pools (p14final, p14final_np2) are refused unless reports/phase14/frozen_p14.json exists.
"""
import json, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase14 import bench14 as B
from edge.phase14.pipeline14 import P14Config, SessionContext, run_event

pool = sys.argv[1]
if pool.startswith("p14final") and not Path("reports/phase14/frozen_p14.json").exists():
    raise SystemExit("final pools require reports/phase14/frozen_p14.json")
fr = json.loads(Path("reports/phase14/frozen_p14.json").read_text())["p14"] if Path("reports/phase14/frozen_p14.json").exists() else {}
cfg = P14Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in fr.items()})
if len(sys.argv) > 3:
    cfg.n_cand, cfg.rounds = int(sys.argv[2]), int(sys.argv[3])
ctx = SessionContext(B.noise_bank(B.POOLS[pool]["noise"]), cfg)
cal = B.generate(pool, 150, 0, 0, calibration=True)
t = time.time()
thr2 = ctx.calibrate([e.snippet for e in cal])
ev = B.generate(pool, 150, 60, 60)
rows = []
for e in ev:
    o = run_event(ctx, e.snippet)
    rows.append({"kind": e.kind, "cond": e.cond, "sources": e.sources, "snr": e.snr, "dt": e.dt, "sep": e.sep,
                 "res": e.resolvable, "cells": tuple(map(str, e.cells)), **o})
cal_rows = [{"kind": "single", **run_event(ctx, e.snippet)} for e in cal[:300]]
m0 = next(iter(ctx._models.values()))
tag = f"_c{cfg.n_cand}r{cfg.rounds}" if len(sys.argv) > 3 else ""
pickle.dump({"rows": rows, "cal_rows": cal_rows, "thr2": thr2, "cfg": cfg.__dict__}, open(f"data/phase14/p14rows_{pool}{tag}.pkl", "wb"))
print(pool, tag, "thr2", round(thr2, 3), len(rows), f"{time.time() - t:.0f}s",
      "count ms median", round(float(np.median([r["ms_count"] for r in rows])), 2), flush=True)
