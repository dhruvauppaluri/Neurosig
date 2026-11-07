"""Goal F: per-event latency, memory, channel scaling and burst behaviour. Python wall-clock, single process.

Burst test is a deterministic discrete-event simulation using measured per-event service times (sampled with a fixed
seed) through a bounded FIFO and a single worker, with and without a tiered policy that falls back to the fast
K=1 fit when the queue is deep. Not a hardware latency claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from edge.benchmark_v3 import _array, _engines
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.phase8.pipeline import Calibration, Phase8Localizer
from edge.phase8.refine import RefinedScanLocalizer
from edge.phase8.registry import versions
from edge.phase8.synth import make_events
from edge.snippets import extract_snippet


def engines_for(array, cal):
    e = _engines(array)
    e["refined_scan"] = RefinedScanLocalizer(array)
    e["phase8"] = Phase8Localizer(array, cal)
    return e


def time_events(engine, snippets):
    wall, cpu = [], []
    for sn in snippets:
        t0, c0 = time.perf_counter(), time.process_time()
        engine.estimate(sn)
        wall.append((time.perf_counter() - t0) * 1e3); cpu.append((time.process_time() - c0) * 1e3)
    return np.array(wall), np.array(cpu)


def pct(a):
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def simulate_burst(service_fast, service_full, rate_hz, duration_s=2.0, queue_cap=256, policy="full", depth_switch=16, seed=1):
    rng = np.random.default_rng(seed)
    n = int(rate_hz * duration_s)
    arrivals = np.sort(rng.uniform(0, duration_s, n))        # Poisson-like burst, fixed seed
    t_free, queue, done, drops, delays, tier_fast = 0.0, [], 0, 0, [], 0
    i = 0
    events = list(arrivals)
    pending = []
    now = 0.0
    while i < len(events) or pending:
        if not pending and i < len(events) and now < events[i]:
            now = events[i]
        while i < len(events) and events[i] <= now:
            if len(pending) >= queue_cap:
                drops += 1
            else:
                pending.append(events[i])
            i += 1
        if not pending:
            continue
        a = pending.pop(0)
        use_fast = policy == "tiered" and len(pending) >= depth_switch
        s = (service_fast if use_fast else service_full)[rng.integers(0, len(service_fast))] / 1000.0
        tier_fast += use_fast
        delays.append(now + s - a)
        now += s
        done += 1
    d = np.array(delays) if delays else np.zeros(1)
    return {"offered_hz": rate_hz, "processed": done, "dropped": drops, "drop_rate": drops / max(n, 1),
            "fast_tier_fraction": tier_fast / max(done, 1), "delay_p50_ms": float(np.percentile(d, 50) * 1e3),
            "delay_p99_ms": float(np.percentile(d, 99) * 1e3), "delay_max_ms": float(d.max() * 1e3)}


def main() -> None:
    cal = Calibration(**json.loads(Path("reports/phase8/frozen_calibration.json").read_text()))
    array = _array("linear")
    events = make_events("dev", n_per_cell=70)           # 210 events
    snips = [e.snippet for e in events]
    eng = engines_for(array, cal)
    report = {"versions": versions(), "n_events": len(snips), "per_method": {}, "memory_peak_kib": {}, "scaling": {}}
    lat = {}
    for name, e in eng.items():
        w, c = time_events(e, snips[:200])
        lat[name] = w
        report["per_method"][name] = {**pct(w), "cpu_median_ms": float(np.median(c))}
    for name, e in eng.items():
        tracemalloc.start(); [e.estimate(s) for s in snips[:50]]; report["memory_peak_kib"][name] = tracemalloc.get_traced_memory()[1] / 1024; tracemalloc.stop()
    base = report["per_method"]["robust_scan"]["median_ms"]
    report["ratio_to_robust_scan_median"] = {k: v["median_ms"] / base for k, v in report["per_method"].items()}
    # channel scaling and setup cost
    for nch in (16, 32, 64):
        arr = linear_probe_um(n_channels=nch)
        rows = {}
        sn_list = []
        rng = np.random.default_rng(5)
        for k in range(40):
            truth = (float(rng.uniform(15, 70)), float(rng.uniform(60, nch * 20 - 60)), 0.0)
            tr = render_recording([PointSource(64, truth, 12.0)], arr, 32000.0, 5.0, int(rng.integers(0, 2**31)), 160)
            sn_list.append(extract_snippet(tr, 64, arr, event_id=k, ground_truth_xyz_um=truth))
        for name in ("beamform", "robust_scan", "refined_scan", "phase8"):
            t0 = time.perf_counter()
            if name == "phase8":
                e = Phase8Localizer(arr, cal)
            elif name == "refined_scan":
                e = RefinedScanLocalizer(arr)
            else:
                from edge.localize import get_localizer
                e = get_localizer(name, array=arr)
            setup_ms = (time.perf_counter() - t0) * 1e3
            w, _ = time_events(e, sn_list)
            rows[name] = {"setup_ms": setup_ms, **pct(w)}
        report["scaling"][str(nch)] = rows
    # bursts
    fast, full = lat["refined_scan"], lat["phase8"]
    report["burst"] = {}
    for rate in (50, 100, 300, 1000):
        report["burst"][str(rate)] = {p: simulate_burst(fast, full, rate, policy=p) for p in ("full", "tiered")}
    report["capacity"] = {"phase8_events_per_s_single_core": 1000.0 / report["per_method"]["phase8"]["median_ms"],
                          "refined_events_per_s_single_core": 1000.0 / report["per_method"]["refined_scan"]["median_ms"],
                          "beamform_events_per_s_single_core": 1000.0 / report["per_method"]["beamform"]["median_ms"]}
    report["notes"] = ["No batch speedup measured: per-event fits are scipy-call-overhead bound; matrix-product scans are already O(CG).",
                       "Expected 32-channel event rate is an assumption (order 10-300 events/s), not measured here."]
    Path("reports/phase8/runtime_bench.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"per_method": report["per_method"], "ratio": report["ratio_to_robust_scan_median"],
                      "capacity": report["capacity"]}, indent=1))
    for r, v in report["burst"].items():
        print(r, {p: (round(x["drop_rate"], 3), round(x["delay_p99_ms"], 1), round(x["fast_tier_fraction"], 2)) for p, x in v.items()})
    print({k: {m: round(x["median_ms"], 2) for m, x in v.items()} for k, v in report["scaling"].items()})
    print(report["memory_peak_kib"])


if __name__ == "__main__":
    main()
