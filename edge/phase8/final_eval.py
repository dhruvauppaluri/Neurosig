"""Single-shot Phase 8 synthetic evaluation.

Calibrates on dev (n=150/cell), reports select (already used for selection), then evaluates the untouched final
pools ONCE. Run: PYTHONPATH=. .venv/bin/python -m edge.phase8.final_eval   (writes reports/phase8/final_eval.json
and refuses to overwrite an existing final result unless --force-rerun-final is given).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from edge.benchmark_v3 import _array, _engines
from edge.localize.common import fold_linear_probe_ambiguity
from edge.phase8.evaluate import folded_error_um
from edge.phase8.overlap_bench import evaluate_methods, make_overlap_events, summarize_overlap
from edge.phase8.pipeline import Phase8Localizer, fit_calibration, save_calibration
from edge.phase8.refine import RefinedScanLocalizer
from edge.phase8.registry import bootstrap_median_diff, config_hash, error_stats, versions
from edge.phase8.synth import make_events

OUT = Path("reports/phase8/final_eval.json")
LARGE_UM = 50.0


def run_methods(events, array, cal):
    base = _engines(array)
    refined = RefinedScanLocalizer(array)
    p8 = Phase8Localizer(array, cal)
    out = {}
    for name, eng in base.items():
        out[name] = {"err": [], "ms": [], "conf": []}
    out["refined_scan"] = {"err": [], "ms": [], "conf": []}
    out["phase8"] = {"err": [], "ms": [], "conf": [], "accepted": [], "cover": [], "area": [], "unit_area": [],
                     "n_src": []}
    for ev in events:
        for name, eng in base.items():
            t0 = time.perf_counter(); e = eng.estimate(ev.snippet); ms = (time.perf_counter() - t0) * 1e3
            out[name]["err"].append(folded_error_um(e.xyz_um, ev.truth_xyz_um, ev.snippet.electrode_xyz_um) if e.available else np.nan)
            out[name]["ms"].append(ms); out[name]["conf"].append(e.confidence or 0.0)
        t0 = time.perf_counter(); e = refined.estimate(ev.snippet); ms = (time.perf_counter() - t0) * 1e3
        out["refined_scan"]["err"].append(folded_error_um(e.xyz_um, ev.truth_xyz_um, ev.snippet.electrode_xyz_um) if e.available else np.nan)
        out["refined_scan"]["ms"].append(ms); out["refined_scan"]["conf"].append(e.confidence or 0.0)
        t0 = time.perf_counter(); r = p8.localize(ev.snippet); ms = (time.perf_counter() - t0) * 1e3
        o = out["phase8"]
        o["ms"].append(ms); o["accepted"].append(bool(r.accepted)); o["n_src"].append(r.n_sources); o["conf"].append(r.confidence)
        if r.primary is None:
            o["err"].append(np.nan); o["cover"].append(np.nan); o["area"].append(np.nan); o["unit_area"].append(np.nan)
        else:
            truth = np.array([np.hypot(ev.truth_xyz_um[0], ev.truth_xyz_um[2]), ev.truth_xyz_um[1]])
            d = np.array(r.primary.xyz_um[:2]) - truth
            cov = np.array(r.primary.cov_xy)
            m = float(d @ np.linalg.solve(cov, d))
            q = (cal.q_shared if "shared_waveform_pair" in r.flags else
                 (cal.q_two if r.n_sources == 2 else (cal.q_flagged if r.flags else cal.q_clean)))
            o["err"].append(float(np.linalg.norm(d))); o["cover"].append(float(m <= q))
            o["area"].append(r.primary.region_area_um2); o["unit_area"].append(float(np.pi * np.sqrt(max(np.linalg.det(cov), 0))))
    return {k: {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()} for k, v in out.items()}


def summarize(events, res):
    contam = np.array([e.contaminated for e in events]); noise = np.array([e.noise_uv for e in events])
    subsets = {"all": np.ones(len(events), bool), "single": ~contam, "contaminated": contam,
               "noise2": noise == 2.0, "noise10": noise == 10.0, "noise25": noise == 25.0}
    table = {}
    for name, r in res.items():
        err = r["err"]
        acc = r["accepted"].astype(bool) if "accepted" in r else np.isfinite(err)
        table[name] = {}
        for sname, m in subsets.items():
            e_all = err[m]
            e_acc = err[m & acc]
            table[name][sname] = {
                "n_events": int(m.sum()), "accepted": int((m & acc).sum()), "accepted_fraction": float((m & acc).sum() / m.sum()),
                "accepted_errors": error_stats(e_acc),
                "all_event_errors_if_estimate_always_returned": error_stats(e_all[np.isfinite(e_all)]),
                "large_error_rate_all_events": float(np.sum((e_all > LARGE_UM) & acc[m]) / m.sum()),
                "large_error_rate_accepted": float(np.mean(e_acc > LARGE_UM)) if len(e_acc) else None,
                "runtime_ms_median": float(np.median(r["ms"][m])), "runtime_ms_p95": float(np.percentile(r["ms"][m], 95)),
            }
            if "cover" in r:
                c = r["cover"][m & acc]
                table[name][sname]["region_coverage_accepted"] = float(np.nanmean(c)) if len(c) else None
                table[name][sname]["region_area_um2_median"] = float(np.nanmedian(r["area"][m & acc])) if len(c) else None
                table[name][sname]["false_two_source_rate"] = float(np.mean(r["n_src"][m] >= 2)) if sname == "single" else None
    return table


def risk_coverage(events, res, name, score_key):
    """Error p90 and mean vs fraction of events kept, ranking by the method's own uncertainty/confidence."""
    r = res[name]
    err = r["err"]; ok = np.isfinite(err)
    score = r[score_key]
    order = np.argsort(score if score_key == "unit_area" else -score)
    order = [i for i in order if ok[i]]
    out = []
    for frac in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        k = max(1, int(round(frac * len(order))))
        sel = err[order[:k]]
        out.append({"coverage": frac, "n": int(k), "mean_um": float(sel.mean()), "median_um": float(np.median(sel)),
                    "p90_um": float(np.percentile(sel, 90)), "large_rate": float(np.mean(sel > LARGE_UM))})
    return out


def overlap_section(array, cal, pool, n):
    ev = make_overlap_events(pool, n_per_noise=n)
    base = _engines(array); p8 = Phase8Localizer(array, cal)

    def baseline(name):
        def fn(sn):
            e = base[name].estimate(sn)
            if not e.available:
                return np.zeros((0, 2)), True
            a = fold_linear_probe_ambiguity(e.xyz_um, sn.electrode_xyz_um)
            return np.array([[a[0], a[1]]]), False
        return fn

    def phase8(sn):
        r = p8.localize(sn)
        if r.primary is None:
            return np.zeros((0, 2)), True
        return np.array([s.xyz_um[:2] for s in r.sources]), (not r.accepted)
    methods = {"beamform": baseline("beamform"), "template": baseline("template"),
               "robust_scan": baseline("robust_scan"), "phase8": phase8}
    recs = evaluate_methods(ev, methods)
    return {name: summarize_overlap(ev, r) for name, r in recs.items()}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--force-rerun-final", action="store_true"); a = ap.parse_args()
    if OUT.exists() and not a.force_rerun_final:
        raise SystemExit(f"{OUT} exists: the final pool is evaluated once. Use --force-rerun-final only for a documented new protocol.")
    array = _array("linear")
    dev = make_events("dev", n_per_cell=150)
    cal = fit_calibration(array, dev)
    save_calibration(cal, "reports/phase8/frozen_calibration.json")
    frozen_hash = config_hash(cal.to_json())
    report = {"versions": versions(), "frozen_calibration_hash": frozen_hash, "frozen_calibration": cal.to_json(),
              "metric": "folded (hypot(x,z), y) error in um; large error = >50 um",
              "n_per_cell": 150, "pools": {"dev": "calibration only", "select": "selection (already seen)", "final": "evaluated once"}}
    for pool in ("select", "final"):
        ev = make_events(pool, n_per_cell=150)
        res = run_methods(ev, array, cal)
        report[pool] = {"table": summarize(ev, res),
                        "risk_coverage": {"phase8_by_uncertainty": risk_coverage(ev, res, "phase8", "unit_area"),
                                          "beamform_by_confidence": risk_coverage(ev, res, "beamform", "conf"),
                                          "template_by_confidence": risk_coverage(ev, res, "template", "conf")},
                        "paired_vs_best_baseline": {}}
        accepted = res["phase8"]["accepted"].astype(bool)
        for b in ("beamform", "template", "amplitude"):
            m = accepted & np.isfinite(res[b]["err"])
            report[pool]["paired_vs_best_baseline"][b] = bootstrap_median_diff(res[b]["err"][m], res["phase8"]["err"][m])
        report[pool]["overlap"] = overlap_section(array, cal, pool, 40)
        if pool == "final":
            mm = make_events("final", n_per_cell=60, render_model="line_source")
            res_m = run_methods(mm, array, cal)
            report["final_model_mismatch_line_source"] = summarize(mm, res_m)
        print(pool, "done", flush=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    p = report["final"]["table"]["phase8"]["all"]
    print("final phase8 all:", p["accepted_fraction"], p["accepted_errors"], p["region_coverage_accepted"])


if __name__ == "__main__":
    main()
