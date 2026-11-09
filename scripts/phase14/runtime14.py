"""Phase 14.7 runtime: per-event latency, memory and sustained throughput of the frozen P14 event path
(count K<=3 + position + region map) with 1 / 2 / 4 worker processes, against measured detector event rates.

Events: p14select bench snippets (same snippet geometry as real events). Every event is fully processed (no
skipping). Event-rate reference: detector-v2.0 events per second of recording in every cached IBL session
(data/phase14/det_*.npz, 2 s windows). Online association cost is measured separately (exp14_4 ms_per_event).
Writes reports/phase14/exp14_7_runtime.json."""
import json, sys, time, tracemalloc
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase14 import bench14 as B
from edge.phase14.pipeline14 import P14Config, SessionContext, run_event

FR = json.loads(Path("reports/phase14/frozen_p14.json").read_text())
CFG = P14Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in FR["p14"].items()})
_ctx = None


def _init():
    global _ctx
    _ctx = SessionContext(B.noise_bank("select"), CFG); _ctx.thr2 = 4.3


def _work(sn):
    t = time.perf_counter(); run_event(_ctx, sn); return (time.perf_counter() - t) * 1e3


def main():
    ev = [e.snippet for e in B.generate("p14select", 150, 60, 60)] * 2
    rates = {}
    for p in sorted(Path("data/phase14").glob("det_*.npz")):
        z = np.load(p); n_w = len(np.unique(z["win"])); rates[p.stem[4:]] = float(len(z["t"]) / (2.0 * max(n_w, 1)))
    out = {"frozen_hash": FR["hash"], "event_rates_per_s": rates, "max_rate": max(rates.values()), "median_rate": float(np.median(list(rates.values())))}
    # single-process latency + memory
    _init()
    for sn in ev[:50]:
        run_event(_ctx, sn)
    lat = [_work(sn) for sn in ev[:1000]]
    tracemalloc.start(); [run_event(_ctx, sn) for sn in ev[:200]]; cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["latency_ms"] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)), "max": float(np.max(lat))}
    out["peak_traced_memory_mib"] = peak / 2 ** 20
    out["throughput"] = {}
    for w in (1, 2, 4):
        with Pool(w, initializer=_init) as p:
            p.map(_work, ev[:40 * w])
            t = time.perf_counter(); L = p.map(_work, ev, chunksize=16); dt = time.perf_counter() - t
        thr = len(ev) / dt
        out["throughput"][str(w)] = {"events_per_s": thr, "latency_median_ms": float(np.median(L)), "latency_p99_ms": float(np.percentile(L, 99)),
                                     "margin_vs_max_rate": thr / out["max_rate"], "margin_vs_median_rate": thr / out["median_rate"]}
        print(w, out["throughput"][str(w)], flush=True)
    Path("reports/phase14/exp14_7_runtime.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "event_rates_per_s"}, indent=1))


if __name__ == "__main__":
    main()
