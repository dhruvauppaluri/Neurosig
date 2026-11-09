"""Phase 14.2-14.3 feature harness: for every event of a Phase 14 bench pool, run the P12 baseline (P12-LR count
probability on the frozen Phase 9/10 K<=2 fit) and the Phase 14 single-neuron null families (line20 / ms / ms5,
K <= 3), and store all statistics so every decision rule is scored offline on identical fits.

Usage: .venv/bin/python scripts/phase14/run_14_2.py <pool> [scale] [workers]
Output (git-ignored): data/phase14/rows_<pool>.pkl
"""
import json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
SPECS = ("line20", "ms", "ms5", "msq", "msn", "msw", "ms3")
DIAG = ("line20", "ms", "ms5")
_cache = {}


def _models(E):
    key = E.tobytes()
    if key not in _cache:
        from edge.phase14.neuron_model import SourceModel
        from edge.phase12.fast import fast_localizer
        from edge.phase12.overlap_bench import _array
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = ({s: SourceModel(E, s) for s in SPECS}, fast_localizer(_array(E)))
    return _cache[key]


_noise = {}


def noise_model(pool):
    if pool not in _noise:
        from edge.phase14 import bench14 as B
        from edge.phase14.neuron_model import NoiseModel
        _noise[pool] = NoiseModel(B.noise_bank(B.POOLS[pool]["noise"]))
    return _noise[pool]


def work(ev):
    from edge.phase12 import count_model as CM
    from edge.phase14.neuron_model import prepare
    lr = CM.LogisticCount.from_json(json.loads(Path("reports/phase12/count_model_p12.json").read_text())["logistic"])
    E = np.asarray(ev.snippet.electrode_xyz_um, float)
    models, loc = _models(E)
    r = {"kind": ev.kind, "cond": ev.cond, "sources": ev.sources, "snr": ev.snr, "dt": ev.dt, "sep": ev.sep,
         "res": ev.resolvable, "cells": tuple(map(str, ev.cells)), "tids": ev.tids, "E": E[:, :2]}
    t = time.perf_counter()
    try:
        a = CM.analyze_fast(loc, ev.snippet)
    except Exception as e:  # noqa: BLE001
        a = None; r["lr_error"] = str(e)[:80]
    r["ms_lr"] = (time.perf_counter() - t) * 1e3
    if a is not None:
        r.update(lr_prob=float(lr.prob(a["x"])[0]), lr_thr=lr.thr, lr_pos1=a["pos1"], lr_pos2=a["pos2"],
                 rank2=a["rank2"], n_trough=a["n_trough"], t_spread=a["t_spread"], lr_x=a["x"],
                 lr_two=bool(lr.prob(a["x"])[0] > lr.thr))
        d = CM.decide_lr(a, __import__("edge.phase10.overlap_stress", fromlist=["x"]).frozen_calibration(), lr)
        r["lr_n"] = d.n; r["lr_pos"] = d.pos
    pr = prepare(ev.snippet)
    r["snr_obs"] = float(np.max(-pr.D.min(axis=1))) if pr.D.size else 0.0
    nm = noise_model(ev.pool)
    variants = {"": pr, "_st": prepare(ev.snippet, noise=nm), "_sp": prepare(ev.snippet, noise=nm, temporal=False)}
    for s, m in models.items():
        for suf, p in variants.items():
            if (suf == "" and s not in DIAG) or (suf == "_st" and s not in ("ms", "ms5")):
                continue
            t = time.perf_counter()
            f = m.fit(p, 3)
            f["ms"] = (time.perf_counter() - t) * 1e3
            r[s + suf] = f
    return r


def main():
    pool = sys.argv[1]; cal = pool.startswith("cal:"); pool = pool.split(":")[-1]; scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    from edge.phase14 import bench14 as B
    ev = B.generate(pool, int(150 * scale), int(60 * scale), int(60 * scale), calibration=cal)
    t = time.time()
    with Pool(nw) as p:
        rows = p.map(work, ev, chunksize=8)
    out = Path(f"data/phase14/rows_{'cal_' if cal else ''}{pool}.pkl")
    pickle.dump(rows, open(out, "wb"))
    print(pool, len(rows), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
