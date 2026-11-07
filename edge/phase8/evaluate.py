"""Shared evaluation helpers: run engines on identical events and summarize by subset."""

from __future__ import annotations

import time

import numpy as np

from edge.phase8.registry import error_stats


def folded_error_um(xyz, truth, electrodes) -> float:
    """Distance after mapping both points onto the identifiable half-plane when the array is collinear.

    A linear probe cannot separate x from z (rotation about the shank), so truth (x, y, z) is compared as
    (hypot(x, z), y, 0). Arrays with lateral span are scored in 3D.
    """
    from edge.localize.common import fold_linear_probe_ambiguity
    a = np.asarray(fold_linear_probe_ambiguity(tuple(xyz), electrodes))
    b = np.asarray(fold_linear_probe_ambiguity(tuple(truth), electrodes))
    return float(np.linalg.norm(a - b))


def run_engines(engines: dict, events) -> dict[str, dict]:
    """Return {name: {'error': array (nan=rejected), 'runtime_ms': array}} aligned with events."""
    out = {}
    for name, eng in engines.items():
        err, rt = [], []
        for ev in events:
            t0 = time.perf_counter()
            est = eng.estimate(ev.snippet)
            rt.append((time.perf_counter() - t0) * 1000)
            err.append(folded_error_um(est.xyz_um, ev.truth_xyz_um, ev.snippet.electrode_xyz_um)
                       if est.available and est.xyz_um is not None else np.nan)
        out[name] = {"error": np.asarray(err), "runtime_ms": np.asarray(rt)}
    return out


def summarize(res: dict, events, subsets: dict | None = None) -> dict:
    subsets = subsets or {}
    masks = {"all": np.ones(len(events), bool)}
    masks.update({k: np.asarray([f(e) for e in events]) for k, f in subsets.items()})
    table = {}
    for name, r in res.items():
        table[name] = {}
        for k, m in masks.items():
            e = r["error"][m]
            s = error_stats(e[np.isfinite(e)])
            s["rejected"] = int(np.sum(~np.isfinite(e)))
            s["total"] = int(m.sum())
            s["runtime_ms_median"] = float(np.median(r["runtime_ms"][m])) if m.any() else None
            table[name][k] = s
    return table


def fmt(table: dict, subset: str = "all") -> str:
    lines = [f"{'method':16s} {'n':>4s} {'rej':>4s} {'med':>6s} {'p90':>6s} {'p95':>6s} {'mean':>6s} {'worst':>6s} {'ms':>6s}"]
    for name, t in table.items():
        s = t[subset]
        if s.get("n", 0) == 0:
            lines.append(f"{name:16s} none")
            continue
        lines.append(f"{name:16s} {s['n']:4d} {s['rejected']:4d} {s['median']:6.1f} {s['p90']:6.1f} "
                     f"{s['p95']:6.1f} {s['mean']:6.1f} {s['worst']:6.1f} {s['runtime_ms_median']:6.2f}")
    return "\n".join(lines)
