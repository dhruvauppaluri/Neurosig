"""Phase 1: corrected real-data localization baseline for all five existing methods on identical events.

  .venv/bin/python scripts/phase9/baseline_corrected.py            # pilot reproduction + all cached windows
Writes reports/phase9/corrected_baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge import ibl_eval  # noqa: E402
from edge.localize import LOCALIZATION_METHODS  # noqa: E402
from edge.phase8.registry import versions  # noqa: E402
from edge.phase9.events import sorter_snippets, trough_offset_ms, window_sorter_seconds  # noqa: E402
from scripts.phase5.data import load_window, window_path  # noqa: E402
from scripts.real_detector import frozen_real_detector, preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PILOT_PUBLISHED = {"amplitude": 21.7, "template": 20.0, "beamform": 19.4, "robust_scan": 22.6}
PILOT_TOL_UM = 0.5   # corrected pilot is deterministic; allow rounding in the published table (1 decimal)


def pilot_engines(array):
    """Engine configuration of edge.ibl.run_pilot (robust_scan keeps the default offset_reciprocal model)."""
    from edge.localize import get_localizer
    from edge.localize.beamform import default_scan_grid_um
    from edge.localize.template import default_training_xyz_um
    y_shift = float(np.min(array.xyz_um()[:, 1]))
    training = default_training_xyz_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    scan = default_scan_grid_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    return {"amplitude": get_localizer("amplitude"),
            "template": get_localizer("template", array=array, training_xyz_um=training),
            "beamform": get_localizer("beamform", array=array, grid_xyz_um=scan),
            "weighted_scan": get_localizer("weighted_scan", array=array, grid_xyz_um=scan),
            "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan)}


def run_methods(array, snippets, meta, engines=None):
    eng = engines if engines is not None else ibl_eval._engines(array)
    rows = {m: [] for m in LOCALIZATION_METHODS}
    for sn, mt in zip(snippets, meta):
        for m in LOCALIZATION_METHODS:
            t0 = time.perf_counter()
            e = eng[m].estimate(sn)
            rt = (time.perf_counter() - t0) * 1e3
            ok = bool(e.available and e.xyz_um is not None)
            rows[m].append({"ok": ok, "xyz": list(e.xyz_um) if ok else None, "ms": rt,
                            "axial": abs(e.xyz_um[1] - mt["ref_y_um"]) if ok else None,
                            "cluster": mt["cluster"], "pid": mt["pid"]})
    return rows


def stats(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95))}


def unit_spread(rows):
    by = {}
    for r in rows:
        if r["ok"]:
            by.setdefault((r["pid"], r["cluster"]), []).append(r["xyz"][:2])
    sp = []
    for v in by.values():
        if len(v) >= 5:
            a = np.asarray(v)
            sp.append(float(np.median(np.linalg.norm(a - np.median(a, axis=0), axis=1))))
    return {"n_units": len(sp), "median_um": float(np.median(sp)) if sp else None,
            "p90_um": float(np.percentile(sp, 90)) if sp else None}


def summarize(rows, snippets):
    out = {}
    for m, rr in rows.items():
        out[m] = {"axial_disagreement_um": stats([r["axial"] for r in rr]),
                  "valid_fraction": float(np.mean([r["ok"] for r in rr])),
                  "runtime_ms_median": float(np.median([r["ms"] for r in rr])),
                  "within_unit_spread_xy_um": unit_spread(rr)}
    names = list(rows)
    dis = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = [np.linalg.norm(np.subtract(x["xyz"], y["xyz"])) for x, y in zip(rows[a], rows[b]) if x["ok"] and y["ok"]]
            dis[f"{a}|{b}"] = float(np.median(d)) if d else None
    out["method_disagreement_median_um"] = dis
    return out


def alignment(w, clock, max_events=300):
    cfg = frozen_real_detector(w["fs"])
    filt, _ = preprocess(w["voltages_uv"], cfg)
    t = window_sorter_seconds(w, clock)
    s = np.round(t * w["fs"]).astype(int)
    keep = (s > 90) & (s < filt.shape[0] - 90)
    if keep.sum() > max_events:
        keep = np.flatnonzero(keep)[:max_events]
    else:
        keep = np.flatnonzero(keep)
    off = trough_offset_ms(filt, s[keep], np.asarray(w["sorter_local_channel"])[keep], w["fs"])
    return off


def pilot_reproduction():
    w = load_window(window_path("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 100.0, 1.0))
    res = {}
    for clock in ("session", "probe"):
        array, sn, mt = sorter_snippets(w, clock, max_events=None)
        sn, mt = sn[:50], mt[:50]
        res[clock] = summarize(run_methods(array, sn, mt, pilot_engines(array)), sn)
    ref = json.load(open(ROOT / "reports/phase6/corrected_history/corrected_summary.json")) if (ROOT / "reports/phase6/corrected_history/corrected_summary.json").exists() else None
    out = {"note": "first 50 sorter spikes of 4836a465 t0=100 s, raw (unfiltered) voltages, engine config of edge.ibl.run_pilot (robust_scan: offset_reciprocal). edge.ibl_eval uses line_source for robust_scan, which gives a different pilot value; both are tabulated in aggregate/ibl_eval runs.",
           "session_clock_control": {m: res["session"][m]["axial_disagreement_um"]["median"] for m in LOCALIZATION_METHODS},
           "probe_clock_corrected": {m: res["probe"][m]["axial_disagreement_um"]["median"] for m in LOCALIZATION_METHODS},
           "published_corrected_phase6": PILOT_PUBLISHED}
    out["max_abs_diff_vs_published_um"] = {m: abs(out["probe_clock_corrected"][m] - PILOT_PUBLISHED[m]) for m in PILOT_PUBLISHED}
    out["reproduced_within_tolerance"] = all(v <= PILOT_TOL_UM for v in out["max_abs_diff_vs_published_um"].values())
    out["tolerance_um"] = PILOT_TOL_UM
    out["full_metrics_corrected"] = res["probe"]
    return out


def main():
    manifest5 = json.loads((ROOT / "reports/phase5/split_manifest.json").read_text())["windows"]
    manifest6 = json.loads((ROOT / "reports/phase6/split_manifest_phase6.json").read_text())["windows"]
    wins = [(w["pid"], w["t0_s"], w["duration_s"], w["split"]) for w in manifest5 + manifest6]
    report = {"versions": versions(), "pilot": pilot_reproduction(), "windows": [], "aligned": {}}
    all_rows = {m: [] for m in LOCALIZATION_METHODS}
    all_sn, align = [], {"probe": [], "session": []}
    for pid, t0, dur, split in wins:
        p = window_path(pid, t0, dur)
        if not p.exists():
            report["windows"].append({"pid": pid[:8], "t0_s": t0, "status": "missing_dataset"})
            continue
        w = load_window(p)
        array, sn, mt = sorter_snippets(w, "probe", max_events=60, seed=int(t0))
        rows = run_methods(array, sn, mt)
        for m in rows:
            all_rows[m] += rows[m]
        all_sn += sn
        for clock in ("probe", "session"):
            align[clock].append(alignment(w, clock))
        report["windows"].append({"pid": pid[:8], "t0_s": t0, "split": split, "n_events": len(sn)})
    report["aggregate_all_cached_windows"] = summarize(all_rows, all_sn)
    report["n_windows_evaluated"] = sum(1 for x in report["windows"] if x.get("n_events"))
    for clock, arrs in align.items():
        a = np.concatenate(arrs)
        a = a[np.isfinite(a)]
        report["aligned"][clock] = {"n": int(a.size), "median_offset_ms": float(np.median(a)),
                                    "median_abs_offset_ms": float(np.median(np.abs(a))),
                                    "p90_abs_offset_ms": float(np.percentile(np.abs(a), 90)),
                                    "frac_within_0.2ms": float(np.mean(np.abs(a) <= 0.2))}
    out = ROOT / "reports/phase9/corrected_baseline.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({"pilot": {k: report["pilot"][k] for k in ("session_clock_control", "probe_clock_corrected", "published_corrected_phase6", "reproduced_within_tolerance")},
                      "aligned": report["aligned"], "n_windows": report["n_windows_evaluated"],
                      "aggregate": {m: report["aggregate_all_cached_windows"][m] for m in LOCALIZATION_METHODS}}, indent=1, default=float))


if __name__ == "__main__":
    main()
