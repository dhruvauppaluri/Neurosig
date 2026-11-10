"""Phase 15 multi-objective dashboard: one machine-readable row per metric (never a single combined score).

Each row: track, metric, value, ci (if any), units, dataset, split, coverage / denominator note, version, source file,
higher_is_better, stretch_target. ``baseline_rows()`` reads the Phase 14 single-use results; Phase 15 rows are appended
by the experiments with the same schema (``row()``), so the two versions can be compared metric by metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def row(track, metric, value, units, dataset, split, version, source, higher=True, ci=None, coverage=None, target=None, note=None):
    return {"track": track, "metric": metric, "value": None if value is None else float(value), "ci": ci, "units": units,
            "dataset": dataset, "split": split, "coverage": coverage, "version": version, "source": source,
            "higher_is_better": higher, "stretch_target": target, "note": note}


def _j(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def baseline_rows() -> list[dict]:
    R = []
    V = "P14 (frozen 9d8006e97d8f)"
    for pool, geo in (("p14final", "NP1"), ("p14final_np2", "NP2")):
        e = _j(f"reports/phase14/p14_eval_{pool}.json")
        if not e:
            continue
        src = f"reports/phase14/p14_eval_{pool}.json"; ds = f"MEArec unseen morphologies {geo} + IBL T-W noise"
        R += [row("B_overlap", f"false_two_rate_{geo}", e["false_two"]["rate"], "fraction", ds, "final14", V, src, False, e["false_two"]["ci"], "all 900 singles", 0.02),
              row("B_overlap", f"resolvable_recall_{geo}", e["pairs_resolvable"]["recall"], "fraction", ds, "final14", V, src, True, e["pairs_resolvable"]["ci"], "resolvable pairs", 0.80),
              row("B_overlap", f"simultaneous_recall_{geo}", e["pairs_simultaneous_res"]["recall"], "fraction", ds, "final14", V, src, True, e["pairs_simultaneous_res"]["ci"], "dt=0 resolvable", 0.75),
              row("B_overlap", f"unequal_pair_recall_{geo}", e["recall_by_cond"]["unequal"]["all"], "fraction", ds, "final14", V, src, True, None, "unequal pairs (ratio 0.2-0.5)"),
              row("B_overlap", f"triple_recovery_{geo}", e["triples"]["recovered3"], "fraction", ds, "final14", V, src, True),
              row("B_overlap", f"pair_position_error_{geo}", e["pair_err_median_um"], "um", ds, "final14", V, src, False, None, "recovered resolvable pairs"),
              row("B_overlap", f"p_overlap_ece_{geo}", e["p_overlap_ece"], "ECE", ds, "final14", V, src, False),
              row("A_localization", f"single_error_median_{geo}", e["single_err_median_um"], "um (soma)", ds, "final14", V, src, False, None, "all singles"),
              row("A_localization", f"single_error_p90_{geo}", e["single_err_p90_um"], "um (soma)", ds, "final14", V, src, False),
              row("A_localization", f"primary_disc_coverage_{geo}", e["discs"]["normalized"]["coverage_all"], "fraction", ds, "final14", V, src, True, e["discs"]["normalized"]["coverage_ci"], "complete population", 0.90),
              row("A_localization", f"primary_disc_coverage_low_snr_{geo}", e["discs"]["normalized"]["by_snr"].get("0-6"), "fraction", ds, "final14", V, src, True, None, "SNR < 6", 0.90),
              row("A_localization", f"disc_area_{geo}", e["discs"]["normalized"]["median_area_um2"], "um^2", ds, "final14", V, src, False)]
    o = _j("reports/phase14/exp14_4_final_oup14.json")
    if o:
        s = o["summary"]; src = "reports/phase14/exp14_4_final_oup14.json"; ds = "IBL T-W detector events (sorter labels for evaluation only)"
        R += [row("C_identity", "precision_hq", s["precision_hq"]["median"], "fraction", ds, "final14", V, src, True, None, "HQ subset", 0.98),
              row("C_identity", "coverage_hq", s["coverage_hq"]["median"], "fraction", ds, "final14", V, src, True, None, "HQ subset", 0.85),
              row("C_identity", "coverage_all_events", s["coverage_all"]["median"], "fraction", ds, "final14", V, src, True, None, "all detector events", 0.40),
              row("C_identity", "precision_all_matched", s["precision"]["median"], "fraction", ds, "final14", V, src, True),
              row("C_identity", "merge_rate", s["merge_rate"]["median"], "fraction", ds, "final14", V, src, False),
              row("C_identity", "fragmentation", s["fragmentation_mean"]["median"], "units per sorter unit", ds, "final14", V, src, False),
              row("A_localization", "unit_vs_event_spread_reduction", s["spread_reduction"]["median"], "fraction", ds, "final14", V, src, True, None, None, 0.80),
              row("F_runtime", "identity_ms_per_event", s["ms_per_event"]["median"], "ms", ds, "final14", V, src, False)]
    d = _j("reports/phase14/exp14_5_p14final.json")
    if d:
        for k, v in d["summary"].items():
            R.append(row("C_identity", f"drift_tracking_error_{k}", v["tracking_median_um"]["kalman_q2.0_r15.0"], "um", "drift streams, unseen morphologies", "final14", V, "reports/phase14/exp14_5_p14final.json", False))
            R.append(row("C_identity", f"drift_fragmentation_{k}", v["fragmentation"], "units per neuron", "drift streams", "final14", V, "reports/phase14/exp14_5_p14final.json", False))
    rt = _j("reports/phase14/exp14_7_runtime.json")
    if rt:
        src = "reports/phase14/exp14_7_runtime.json"
        R += [row("F_runtime", "event_latency_median", rt["latency_ms"]["median"], "ms", "p14select snippets", "bench", V, src, False, None, None, 1.0),
              row("F_runtime", "event_latency_p95", rt["latency_ms"]["p95"], "ms", "p14select snippets", "bench", V, src, False),
              row("F_runtime", "event_latency_p99", rt["latency_ms"]["p99"], "ms", "p14select snippets", "bench", V, src, False),
              row("F_runtime", "throughput_1_worker", rt["throughput"]["1"]["events_per_s"], "events/s", "p14select snippets", "bench", V, src, True),
              row("F_runtime", "throughput_2_workers", rt["throughput"]["2"]["events_per_s"], "events/s", "p14select snippets", "bench", V, src, True, None, None, 1500),
              row("F_runtime", "peak_memory", rt["peak_traced_memory_mib"], "MiB", "p14select snippets", "bench", V, src, False)]
    c = _j("reports/phase14/vis14_cross_final.json")
    if c:
        tgt = {"SG_sf5": 0.60, "SG_orientation6": 0.40, "DG_orientation8": 0.40, "GB_x": 0.35, "GB_y": 0.35, "NS_image118": 0.15}
        for t, r in c["tasks"].items():
            if r:
                vals = [v["zscore"] for v in r.values()]
                R.append(row("D_vision", f"cross_animal_{t}", float(np.mean(vals)), "balanced accuracy", "Allen 6 final mice (Phase 14)", "final14", V,
                             "reports/phase14/vis14_cross_final.json", True, [float(min(vals)), float(max(vals))], "per-mouse min-max in ci", tgt.get(t),
                             f"chance {list(r.values())[0]['chance']:.3f}"))
    w = _j("reports/phase14/vis14_within_final.json")
    if w:
        for t, s in w["summary"].items():
            b = s["balanced_accuracy_mean"]
            R.append(row("D_vision", f"within_animal_{t}_tb50", b.get("lr:tb50"), "balanced accuracy", "Allen 6 final mice", "final14", V, "reports/phase14/vis14_within_final.json", True))
        for o in w["sessions"].values():
            pass
        u = [o["unseen_combo"]["tb25"] for o in w["sessions"].values()]
        R.append(row("D_vision", "unseen_combo_orientation_error", float(np.median([x["orientation_median_err_deg"] for x in u])), "deg", "Allen 6 final mice", "final14", V, "reports/phase14/vis14_within_final.json", False))
        R.append(row("D_vision", "unseen_combo_logsf_r2", float(np.median([x["logsf_r2"] for x in u])), "R2", "Allen 6 final mice", "final14", V, "reports/phase14/vis14_within_final.json", True))
    i = _j("reports/phase14/ibl14_final.json")
    if i and "T3_contrast5" in i["summary"]:
        b = i["summary"]["T3_contrast5"]["balanced_accuracy_mean"]
        R.append(row("D_vision", "ibl_contrast5_tcounts", b.get("lr:tcounts"), "balanced accuracy", "IBL T-W", "final14", V, "reports/phase14/ibl14_final.json", True, None, None, None, "chance 0.20"))
    return R


def write(rows: list[dict], path):
    Path(path).write_text(json.dumps({"rows": rows, "n": len(rows)}, indent=1, default=float) + "\n")
