"""Phase 10 runtime and memory benchmark on the real 32-contact IBL geometry with real-like synthetic snippets.

Per engine (legacy raw, corrected B, P9, and the Phase 9 K<=2 overlap pipeline):
  latency median / p95 / p99 (wall), CPU median, engine construction time and resident size of the constructed
  engine (tracemalloc, separate pass), and peak transient memory per event (tracemalloc, separate pass; timing runs
  never have tracemalloc active). Single-threaded, this machine; numbers are measurements, not targets.
"""
import json, platform, sys, time, tracemalloc
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9.configs import corrected_baseline_factory, legacy_factory, p9_factory
from edge.phase9.improved import condition
from edge.phase9.synth_real import make_events
from edge.phase10.configs import p10_factory
from edge.phase10.overlap_stress import OV_COND, localizer

array, evs = make_events("dev", n_per_noise=70, contaminated_fraction=0.5)
snips = [e.snippet for e in evs]


class Overlap:
    def __init__(self, array):
        self.loc = localizer(array)

    def estimate(self, sn):
        return self.loc.localize(condition(sn, **OV_COND))


facs = {"legacy_raw": legacy_factory, "corrected_B": corrected_baseline_factory, "P9": p9_factory, "P10": p10_factory,
        "P9_overlap_K2": lambda a: {"overlap_pipeline": Overlap(a)}}
rep = {"versions": versions(), "machine": platform.platform(), "processor": platform.processor(), "n_events": len(snips),
       "geometry": "32-contact IBL selection (dev window), real-like synthetic snippets", "results": {}}
for fk, f in facs.items():
    t = time.perf_counter(); eng = f(array); build_ms = (time.perf_counter() - t) * 1e3
    tracemalloc.start(); eng2 = f(array); resident = tracemalloc.get_traced_memory()[0] / 1024; tracemalloc.stop(); del eng2
    for name, e in eng.items():
        for s in snips[:5]:
            e.estimate(s)                                   # warm-up
        wall, cpu = [], []
        for s in snips:
            w0, c0 = time.perf_counter(), time.process_time()
            e.estimate(s)
            wall.append((time.perf_counter() - w0) * 1e3); cpu.append((time.process_time() - c0) * 1e3)
        peaks = []
        for s in snips[:60]:
            tracemalloc.start(); e.estimate(s); peaks.append(tracemalloc.get_traced_memory()[1] / 1024); tracemalloc.stop()
        w = np.array(wall)
        rep["results"][f"{fk}/{name}"] = {"median_ms": float(np.median(w)), "p95_ms": float(np.percentile(w, 95)),
                                          "p99_ms": float(np.percentile(w, 99)), "cpu_median_ms": float(np.median(cpu)),
                                          "events_per_s_single_core": float(1000 / np.median(w)),
                                          "peak_transient_kib_median": float(np.median(peaks)),
                                          "peak_transient_kib_max": float(np.max(peaks)),
                                          "factory_build_ms": build_ms, "factory_resident_kib": resident}
    print(fk, "done", flush=True)
# P10 dead-contact path: 8 zeroed contacts (one mask -> one cached rebuild); first call includes the engine build
from edge.types import Snippet
dead = []
for s in snips[:100]:
    v = np.asarray(s.voltages_uv, float).copy(); v[:, 4:12] = 0.0
    dead.append(Snippet(s.event_id, s.t_sample, s.t0_sample, s.sample_rate_hz, s.channel_ids, v, s.electrode_xyz_um,
                        s.peak_channel, s.ground_truth_xyz_um))
for name, e in p10_factory(array).items():
    t = time.perf_counter(); e.estimate(dead[0]); first = (time.perf_counter() - t) * 1e3
    w = []
    for s in dead[1:]:
        t = time.perf_counter(); e.estimate(s); w.append((time.perf_counter() - t) * 1e3)
    rep["results"][f"P10_dead8/{name}"] = {"first_call_ms_incl_build": first, "median_ms": float(np.median(w)),
                                           "p95_ms": float(np.percentile(w, 95)), "p99_ms": float(np.percentile(w, 99)),
                                           "peak_transient_kib_median": float("nan"), "factory_resident_kib": float("nan")}
for k, v in rep["results"].items():
    print(f"{k:32s} {v['median_ms']:7.2f} ms p99 {v['p99_ms']:7.2f}  peak {v['peak_transient_kib_median']:8.1f} KiB  "
          f"factory {v['factory_resident_kib']:8.0f} KiB")
Path("reports/phase10").mkdir(parents=True, exist_ok=True)
Path("reports/phase10/runtime_memory.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
