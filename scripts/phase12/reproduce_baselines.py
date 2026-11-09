"""Phase 12 step 1: reproduce the Phase 11 baselines this phase builds on, before any change.

(a) 11A matched-event within-unit spread (weighted_scan, strategies A/B/C) on the development windows, same call as
    scripts/phase11/run_11a.py dev; (b) 11E OV-A source-count metrics on the cached p11select pool; (c) single-source
    and K<=2 runtime on the same machine. Stored values are read from reports/phase11/*.json (never typed).
Writes reports/phase12/baseline_repro.json. Does not modify any earlier report.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9 import harness as H
from edge.phase10.configs import p10_factory
from edge.phase10.overlap_stress import localizer, summarize as osum
from edge.phase11.exp11a import run_windows, summarize
from edge.phase11.overlap import CountRule, evaluate

R11 = Path("reports/phase11")
out = {"versions": versions(), "stored": {}, "reproduced": {}}
t = time.time()
# (a) referencing baseline
recs, ws = run_windows(H.manifest_windows("dev"), p10_factory, align_half=6, log=lambda s: print(s, flush=True))
S = summarize(recs, ["weighted_scan"], ws)
st = json.loads((R11 / "exp11a_dev_K6.json").read_text())["summary"]
for key, src in (("stored", st), ("reproduced", S)):
    m = src["matched"]["engines"]["weighted_scan"]
    out[key]["11A_dev_matched_weighted"] = {s: {"spread_median": m[s]["spread_median"], "dev_p90": m[s]["dev_p90"],
                                                "n_units": m[s]["n_units"]} for s in "ABC"}
    out[key]["11A_dev_events"] = {k: src["events"][k] for k in ("sorter", "detector", "matched")}
pickle.dump({"recs": recs, "wstats": ws}, open("data/phase12/repro_11a_dev.pkl", "wb"))
# (b) overlap baseline (OV-A = P11 rule) on the cached p11select pool
array, sel = pickle.load(open("data/phase11/stress_p11select.pkl", "rb"))
loc = localizer(array)
rows = {f: [evaluate(loc, e, CountRule(axial_only_boundary=True)) for e in sel[f]] for f in sel}
pairs = rows["dt"] + rows["sep"]; res = [r for r in pairs if r["res"]]
out["reproduced"]["11E_select_OV-A"] = {"res_recall": osum(res, 2)["all_sources_recovered"],
                                         "false_two_singles": float(np.mean([r["n_est"] >= 2 for r in rows["single"]])),
                                         "simultaneous_res": osum([r for r in res if r["dt"] == 0], 2)["all_sources_recovered"],
                                         "triples": osum(rows["triple"], 3)["all_sources_recovered"],
                                         "k2_ms_median": float(np.median([r["ms"] for f in rows for r in rows[f]]))}
e = json.loads((R11 / "exp11e_overlap.json").read_text())["select"]["OV-A"]
out["stored"]["11E_select_OV-A"] = {"res_recall": e["resolvable_pairs"]["all_sources_recovered"],
                                     "false_two_singles": e["false_two_singles"],
                                     "simultaneous_res": e["simultaneous_resolvable"]["all_sources_recovered"],
                                     "triples": e["triples"]["all_sources_recovered"],
                                     "k2_ms_median": e["resolvable_pairs"]["runtime_ms_median"]}
# (c) single-source weighted_scan runtime on real dev events (from the run above)
out["reproduced"]["weighted_ms_median"] = float(np.median([r["ms"] for r in recs if r["strategy"] == "A" and not r["null"]]))
out["note"] = ("ms in (c) is the time to run one engine on one snippet inside run_windows; K<=2 ms is the space-time "
               "pipeline per event on the p11select pool. Runtimes are machine-dependent and not expected to match exactly.")
diff = {}
for k in ("11A_dev_matched_weighted",):
    diff[k] = {s: {m: out["reproduced"][k][s][m] - out["stored"][k][s][m] for m in ("spread_median", "dev_p90", "n_units")} for s in "ABC"}
diff["11E_select_OV-A"] = {m: out["reproduced"]["11E_select_OV-A"][m] - out["stored"]["11E_select_OV-A"][m]
                           for m in out["stored"]["11E_select_OV-A"]}
out["difference"] = diff
out["runtime_s"] = time.time() - t
Path("reports/phase12").mkdir(parents=True, exist_ok=True)
Path("reports/phase12/baseline_repro.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(out["difference"], indent=1, default=float))
