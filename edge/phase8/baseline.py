"""Reproduce the v3 324-event synthetic baseline with per-event rows, p95, memory.

Event construction mirrors ``edge.benchmark_v3.run_synthetic`` exactly (same seeds,
RNG draw order, truth/overlap layout). Timing and tracemalloc passes are separate.
Run: .venv/bin/python -m edge.phase8.baseline
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from edge.benchmark_v3 import _array, _engines
from edge.forward import PointSource, render_recording
from edge.localize import LOCALIZATION_METHODS
from edge.phase8.registry import error_stats, versions
from edge.snippets import extract_snippet

SEEDS = (20261103, 20261104, 20261105)
V3_REPORT = Path("reports/localization-benchmark-v3.json")


def v3_events(seeds=SEEDS):
    """Yield (meta, snippet) for the 324 historical events."""
    n = 0
    for geometry in ("linear", "staggered"):
        array = _array(geometry)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for distance in (20.0, 45.0, 75.0):
                for y in (110.0, 310.0, 510.0):
                    truth = (distance + 2.0, y + 7.0, 8.0)
                    for noise in (2.0, 10.0, 25.0):
                        for overlap in (None, 8):
                            sources = [PointSource(64, truth, 12.0)]
                            if overlap is not None:
                                sources.append(PointSource(64 + overlap, (distance + 15.0, y + 55.0, -10.0), 12.0))
                            trace = render_recording(sources, array, 32000.0, noise, int(rng.integers(0, 2**31)), 160)
                            snippet = extract_snippet(trace, 64, array, event_id=n, ground_truth_xyz_um=truth)
                            meta = {"geometry": geometry, "seed": seed, "distance_um": distance, "y_um": y,
                                    "noise_uv": noise, "overlap_samples": overlap, "truth": list(truth)}
                            n += 1
                            yield meta, snippet, array


def run(track_memory: bool = False) -> tuple[list[dict], float]:
    rows, engines = [], {}
    peak_kib = 0.0
    if track_memory:
        tracemalloc.start()
    for meta, snippet, array in v3_events():
        if meta["geometry"] not in engines:
            engines[meta["geometry"]] = _engines(array)
        for name, engine in engines[meta["geometry"]].items():
            t0 = time.perf_counter()
            est = engine.estimate(snippet)
            dt = (time.perf_counter() - t0) * 1000
            rows.append({**meta, "method": name, "available": bool(est.available),
                         "error_um": est.error_um if est.available else None,
                         "xyz_um": list(est.xyz_um) if est.xyz_um else None, "runtime_ms": dt,
                         "reason": est.reason if not est.available else ""})
        if track_memory:
            peak_kib = max(peak_kib, tracemalloc.get_traced_memory()[1] / 1024.0)
    if track_memory:
        tracemalloc.stop()
    return rows, peak_kib


def summarize(rows: list[dict]) -> dict:
    out = {}
    for m in LOCALIZATION_METHODS:
        sel = [r for r in rows if r["method"] == m]
        s = error_stats([r["error_um"] for r in sel])
        rt = np.asarray([r["runtime_ms"] for r in sel])
        s.update({"total": len(sel), "rejected": sum(not r["available"] for r in sel),
                  "rejection_rate": sum(not r["available"] for r in sel) / len(sel),
                  "runtime_ms_median": float(np.median(rt)), "runtime_ms_p95": float(np.percentile(rt, 95))})
        out[m] = s
    return out


def main() -> None:
    rows, _ = run(track_memory=False)
    _, peak = run(track_memory=True)
    summary = summarize(rows)
    stored = json.loads(V3_REPORT.read_text())["synthetic"]["methods"]
    drift = {}
    for m, s in summary.items():
        ref = stored[m]
        drift[m] = {"median_diff_um": s["median"] - ref["median_error_um"],
                    "p90_diff_um": s["p90"] - ref["p90_error_um"],
                    "rejected_diff": s["rejected"] - ref["failures"]}
    report = {"versions": versions(), "seeds": list(SEEDS), "n_events_per_method": len(rows) // len(LOCALIZATION_METHODS),
              "forward_model": "quasi-static monopole offset_reciprocal, lambda=80um, Gaussian white noise 2/10/25 uV",
              "geometries": ["linear 32ch 20um pitch x=z=0", "staggered x=+-10 20um y pitch"],
              "truth_layout": "x=d+2, y=y0+7, z=8; d in 20/45/75, y0 in 110/310/510; 50% add a second source 8 samples later at (d+15, y0+55, -10), scored against source 1",
              "methods": summary, "drift_vs_v3_report": drift, "peak_traced_python_memory_kib": peak,
              "note": "runtime is wall-clock Python; memory from a separate tracemalloc pass"}
    Path("reports/phase8").mkdir(parents=True, exist_ok=True)
    Path("reports/phase8/baseline_v3_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed("reports/phase8/baseline_v3_rows.npz",
                        method=[r["method"] for r in rows],
                        error=[np.nan if r["error_um"] is None else r["error_um"] for r in rows],
                        noise=[r["noise_uv"] for r in rows])
    print(json.dumps({"methods": summary, "drift": drift, "peak_kib": peak}, indent=1))


if __name__ == "__main__":
    main()
