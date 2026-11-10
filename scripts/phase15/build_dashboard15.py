"""Phase 15 multi-objective dashboard: the Phase 14 baseline rows plus Phase 15 rows with the same schema
(edge/phase15/dashboard.row). Missing inputs are skipped, never filled in. Split labels:
  final15      single-use Phase 15 final data (IBL X1-X4, 6 new Allen mice)
  lomo13       leave-one-morphology-out cross-validation over all 13 BBP models (no fresh morphology exists)
  select / dev used development or selection data
Usage: .venv/bin/python scripts/phase15/build_dashboard15.py -> reports/phase15/dashboard_phase15.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase15.dashboard import baseline_rows, row  # noqa: E402

R = Path("reports/phase15")
J = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else None  # noqa: E731
fr = J("frozen_p15.json") or {}
V = f"P15 (frozen {fr.get('hash', 'unfrozen')})"
rows = []

o = J("overlap_lomo_nested_line20-ms3.json")
if o:
    h = o["hgb"]; src = "reports/phase15/overlap_lomo_nested_line20-ms3.json"; ds = "MEArec 13 BBP L5 morphologies NP1 + IBL noise"
    rows += [row("B_overlap", "false_two_rate_NP1", h["single"], "fraction", ds, "lomo13", "B-P15", src, False, None, f"{h['n']['single']} singles; worst morphology {h['max_morph_false_two']:.3f}", 0.02),
             row("B_overlap", "resolvable_recall_NP1", h["res"], "fraction", ds, "lomo13", "B-P15", src, True, None, f"{h['n']['res']} resolvable pairs", 0.80),
             row("B_overlap", "simultaneous_recall_NP1", h["sim"], "fraction", ds, "lomo13", "B-P15", src, True, None, f"{h['n']['sim']} dt=0 pairs", 0.75),
             row("B_overlap", "unequal_pair_recall_NP1", h["unequal"], "fraction", ds, "lomo13", "B-P15", src, True),
             row("B_overlap", "triple_recovery_NP1", h["triple"], "fraction", ds, "lomo13", "B-P15", src, True),
             row("B_overlap", "resolvable_recall_at_matched_2pct_NP1", h["res_matched2"], "fraction", ds, "lomo13", "B-P15", src, True, None, "threshold set to 2% false-two (oracle, comparison only)")]
    f = o["F21"]
    rows += [row("B_overlap", "false_two_rate_NP1", f["single"], "fraction", ds, "lomo13", "P14-C (F21) same protocol", src, False, None, None, 0.02),
             row("B_overlap", "resolvable_recall_NP1", f["res"], "fraction", ds, "lomo13", "P14-C (F21) same protocol", src, True, None, None, 0.80),
             row("B_overlap", "triple_recovery_NP1", f["triple"], "fraction", ds, "lomo13", "P14-C (F21) same protocol", src, True)]

u = J("uncertainty15_lomo.json")
if u:
    src = "reports/phase15/uncertainty15_lomo.json"; ds = "MEArec 13 morphologies, all conditions"
    for est, name in (("d_ens", "A-P15 ensemble"), ("d_ms3", "ms3 (P14 estimator)")):
        e = u["estimators"][est]
        rows += [row("A_localization", "single_error_median", e["median_err_all"], "um (soma)", ds, "lomo13", name, src, False),
                 row("A_localization", "single_error_p90", e["p90_err_all"], "um (soma)", ds, "lomo13", name, src, False),
                 row("A_localization", "disc_coverage_complete_population", e["coverage_all"], "fraction", ds, "lomo13", name, src, True, e["ci"],
                     f"min per morphology {e['min_morphology_coverage']:.3f}", 0.90),
                 row("A_localization", "disc_area", e["median_area_um2"], "um^2", ds, "lomo13", name, src, False)]

for tag, ver in (("final_oup14", "OU-P14 (reference)"), ("final_idp15", "ID-P15"), ("final_long_oup14", "OU-P14 (reference), 30 s"), ("final_long_idp15", "ID-P15, 30 s")):
    d = J(f"id_{'final' if 'long' not in tag else 'final_long'}_{tag}.json")
    if d:
        s = d["summary"]; src = f"reports/phase15/id_*_{tag}.json"; ds = "IBL X1-X4 detector events (sorter labels for evaluation only)"
        rows += [row("C_identity", "precision_hq", s["precision_hq"], "fraction", ds, "final15", ver, src, True, None, "median over sessions", 0.98),
                 row("C_identity", "coverage_hq", s["coverage_hq"], "fraction", ds, "final15", ver, src, True, None, "median over sessions", 0.85),
                 row("C_identity", "coverage_all_events", s["coverage_all"], "fraction", ds, "final15", ver, src, True, None, "all detector events", 0.40),
                 row("C_identity", "precision_all_matched", s["precision"], "fraction", ds, "final15", ver, src, True),
                 row("A_localization", "unit_vs_event_spread_reduction", s.get("spread_reduction"), "fraction", ds, "final15", ver, src, True, None, None, 0.80)]

TGT = {"SG_sf5": 0.60, "SG_orientation6": 0.40, "DG_orientation8": 0.40, "GB_x": 0.35, "GB_y": 0.35, "NS_image118": 0.15}
for tag, ver in (("final_dp15", "D-P15"), ("final_ref14", "Phase 14 reference spec")):
    d = J(f"xa_final_{tag}.json")
    if d:
        for t, r in d["tasks"].items():
            if r:
                v = [x["bacc"] for x in r.values()]
                rows.append(row("D_vision", f"cross_animal_{t}", float(np.mean(v)), "balanced accuracy", f"Allen {len(v)} Phase 15 final mice", "final15", ver,
                                f"reports/phase15/xa_final_{tag}.json", True, [float(min(v)), float(max(v))], "per-mouse min-max in ci", TGT.get(t)))

s = J("stream15.json")
if s:
    for r in s:
        if r.get("mode") == "compute":
            rows += [row("F_runtime", "event_compute_latency_median_bs1", r["bs1_ms_median"], "ms", f"IBL session {r['session']} events", "bench", "D-P15 batch path, 1 thread",
                         "reports/phase15/stream15.json", False, None, f"p95 {r['bs1_ms_p95']:.2f}, p99 {r['bs1_ms_p99']:.2f} ms", 1.0),
                     row("F_runtime", "event_compute_ms_per_event_bs32", r["bs32_ms_per_event_median"], "ms", f"IBL session {r['session']} events", "bench", "D-P15 batch path, 1 thread",
                         "reports/phase15/stream15.json", False, None, None, 1.0)]
            continue
        lbl = f"{r['workers']}w x{r['speed']:g} b{r['batch']} ({r['session']})"
        rows += [row("F_runtime", "stream_sustained_events_per_s", r["sustained_events_per_s"], "events/s", "IBL dev session stream", "bench", f"D-P15 {lbl}", "reports/phase15/stream15.json", True,
                     None, f"offered {r['offered_events_per_s']:.0f}/s, drops {r['drops']}", 1500),
                 row("F_runtime", "stream_latency_median", r["latency_ms_median"], "ms", "IBL dev session stream", "bench", f"D-P15 {lbl}", "reports/phase15/stream15.json", False, None, None, 1.0),
                 row("F_runtime", "stream_latency_p99", r["latency_ms_p99"], "ms", "IBL dev session stream", "bench", f"D-P15 {lbl}", "reports/phase15/stream15.json", False)]

b = J("batch_test.json")
if b:
    for pool, r in b.items():
        rows.append(row("F_runtime", "batch_compute_ms_per_event_bs16", r["ms_per_event_bs16"], "ms", f"MEArec {pool} (per-event geometry)", "bench", "D-P15 batch path",
                        "reports/phase15/batch_test.json", False, None, "every simulated event has its own geometry: worst case for batching", 1.0))

rc = J("real_count15_final.json")
if rc:
    sm = rc["summary"]; src = "reports/phase15/real_count15_final.json"; ds = "IBL X1-X3 sorter-isolated vs sorter-coincident events (proxy)"
    rows += [row("B_overlap", "real_isolated_two_rate", sm["isolated_two_bp15"], "fraction", ds, "final15", "B-P15", src, False, None, "proxy, not ground truth"),
             row("B_overlap", "real_isolated_two_rate", sm["isolated_two_p14"], "fraction", ds, "final15", "P14-C", src, False, None, "proxy, not ground truth"),
             row("B_overlap", "real_coincident_two_rate", sm["coincident_two_bp15"], "fraction", ds, "final15", "B-P15", src, True, None, "proxy: sorter-coincident is not a verified overlap"),
             row("B_overlap", "real_coincident_two_rate", sm["coincident_two_p14"], "fraction", ds, "final15", "P14-C", src, True, None, "proxy")]

out = {"baseline_p14": baseline_rows(), "phase15": rows, "frozen_hash": fr.get("hash")}
(R / "dashboard_phase15.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(len(out["baseline_p14"]), "baseline rows,", len(rows), "phase15 rows")
