"""Write reports/phase11/experiment_registry.jsonl from saved result files (numbers are read, never typed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import log_experiment as _log

R = Path("reports/phase11"); REG = R / "experiment_registry.jsonl"
if REG.exists():
    REG.unlink()
J = lambda p: json.loads((R / p).read_text())
def log(e): _log(e, REG)

def mt(path, eng="weighted_scan"):
    m = J(path)["summary"]["matched"]["engines"][eng]
    return {s: {"spread": m[s]["spread_median"], "p90": m[s]["dev_p90"], "null": m[s]["null_spread_median"], "vs_A": m[s]["vs_A"]} for s in "ABC"}

for i, (path, split, hyp, var) in enumerate([
        ("exp11a_dev_K6.json", "dev A-C", "Detector-referenced events (B) are more stable than sorter-referenced (A)", "B"),
        ("exp11a_dev_K6.json", "dev A-C", "Local multichannel alignment on the filtered trace (C, +-0.2 ms) stabilizes sorter events", "C"),
        ("exp11a_dev_K3_raw.json", "dev A-C", "E3: the C rule applied to the engines' own raw-CAR signal (+-3 samples) stabilizes sorter events", "C"),
        ("exp11a_select_K6.json", "selection (Phase 5 held-out + E-H + I-L, previously used)", "Confirm B/C on the selection split", "B,C")]):
    log({"id": f"P11-A{i + 1}", "hypothesis": hyp, "split": split, "params": {"file": path, "variant": var},
         "before": {"A": mt(path)["A"]}, "after": {k: v for k, v in mt(path).items() if k in var},
         "decision": "not adopted as the default reference (median not improved; selection median worse)" if i < 3 else "reported",
         "reasoning": "matched-event within-unit spread median unchanged or worse; p90 tail lower on dev"})
d = J("decompose_11a_dev.json")
log({"id": "P11-A5-decomposition", "hypothesis": "Instability decomposes into timing, SNR and reference effects", "split": "dev A-C",
     "params": {"engine": d["engine"]}, "before": {"observed_dev_by_snr": d["observed_dev_by_snr_um"]},
     "after": {"timing_shift": d["timing_shift_displacement_um"], "snr_degradation": d["snr_degradation_displacement_um"],
               "events_without_trough": d["fraction_sorter_events_without_trough"]},
     "decision": "measurement", "reasoning": "noise explains about half of observed deviation per SNR bin; 2-sample timing shift moves clean events ~12 um"})
b = J("exp11b_dev.json")["loso"]; bc = J("exp11b_dev_line20c.json")["loso"]; bs = J("exp11b_select.json")["models"]; bb = J("exp11b_biophys.json")["models"]
for n in ("linefree", "radial", "aniso", "twoline", "line20c"):
    src = bc if n == "line20c" else b
    log({"id": f"P11-B-{n}", "hypothesis": f"Forward model {n} improves held-out prediction and real-data stability over line20",
         "split": "dev LOSO A-C; selection; MEArec biophysical", "params": {"model": n},
         "before": {"line20": {"heldout": b["line20"]["heldout_contact_rel_err_median"], "spread": b["line20"]["unit_spread_median_um"], "biophys_snr8": bb["line20"]["snr8"]}},
         "after": {"dev": src[n], "select": bs.get(n), "biophys_snr8": bb.get(n, {}).get("snr8")},
         "decision": "rejected (line20 retained)", "reasoning": "held-out reconstruction gains do not translate into within-unit stability; no model meets -15% spread"})
c = J("exp11c_sessions.json")
log({"id": "P11-C-sessions", "hypothesis": "Difficult sessions are explained by reference/signal quality rather than the model",
     "split": "12 historical sessions", "params": {"rule": "categories fixed in run_11c.py"}, "before": {}, "after": {"spearman": c["spearman_vs_full_spread_n12"],
     "good_vs_other": J("exp11c_good_vs_other.json")}, "decision": "taxonomy retained", "reasoning": "IBL good units stable in every session"})
dd = J("exp11d_dropout.json")
log({"id": "P11-D-fallback", "hypothesis": "A disagreement-based fallback repairs P10 dead-contact tails", "split": "dev (choice), selection (check)",
     "params": {"chosen": dd["chosen"]}, "before": {"dev_P10": dd["dev"]["P10|0"], "select_P10": dd["select"]["P10|0"]},
     "after": {"dev": dd["dev"][dd["chosen"]], "select": dd["select"][dd["chosen"]]}, "decision": "retained in P11", "reasoning": "pre-registered rule"})
e = J("exp11e_overlap.json")
for n in ("OV-A", "OV-A+B", "OV-A+B+K3"):
    log({"id": f"P11-E-{n}", "hypothesis": f"Count rule {n} improves source-count selection", "split": "calibrate p11dev, compare p11select",
         "params": e["calibration"], "before": {k: e["select"]["P10"][k] for k in ("false_two_singles",)} | {"res_recall": e["select"]["P10"]["resolvable_pairs"]["all_sources_recovered"]},
         "after": {"false_two": e["select"][n]["false_two_singles"], "res_recall": e["select"][n]["resolvable_pairs"]["all_sources_recovered"],
                   "simultaneous": e["select"][n]["simultaneous_resolvable"]["all_sources_recovered"], "triples": e["select"][n]["triples"]["all_sources_recovered"]},
         "decision": {"OV-A": "retained", "OV-A+B": "rejected (no gain)", "OV-A+B+K3": "K3 experimental, off"}[n], "reasoning": "selection pool"})
f = J("exp11f_calibration.json")["models"]
log({"id": "P11-F-mondrian", "hypothesis": "Conditional (stratum x SNR x dropout) conformal calibration gives ~90% coverage per SNR group",
     "split": "calibrate p11dev, evaluate p11select", "params": {"min_n": 30}, "before": {"phase9_frozen": {k: f["phase9_frozen"][k] for k in ("coverage", "area_median_um2", "false_confidence_rate")}},
     "after": {"mondrian": {k: f["mondrian"][k] for k in ("coverage", "coverage_ci95", "area_median_um2", "false_confidence_rate", "by_snr_g")}},
     "decision": "retained in P11", "reasoning": "coverage near nominal in each SNR group with smaller regions"})
for mode in ("maxres", "greedy2"):
    g = J(f"exp11g_runtime_{mode}.json")
    log({"id": f"P11-G-{mode}", "hypothesis": "A cheap spatial ambiguity gate avoids most K<=2 fits without recall loss", "split": "gate on p11dev, evaluate p11select",
         "params": {"mode": mode, "tau": g["tau_gate"]}, "before": g["full"], "after": g["staged"], "decision": "rejected (invokes ~94% of events)",
         "reasoning": "overlap evidence is largely temporal; single-sample spatial score cannot separate"})
if (R / "final11.json").exists():
    fz = J("final11.json")
    log({"id": "P11-FINAL", "hypothesis": "Frozen P11 on fresh final insertions M-P and the p11final pool", "split": "final (single use)",
         "params": {"p11_hash": fz["p11_hash"]}, "before": {}, "after": {"referencing_matched_weighted": fz["referencing"]["matched"]["engines"]["weighted_scan"],
         "dropout": fz["dropout"], "forward_model": fz["forward_model"], "overlap": {k: {"res": v["resolvable_pairs"]["all_sources_recovered"], "false2": v["false_two_singles"]} for k, v in fz["overlap_p11final"].items()},
         "calibration": {k: fz["calibration_p11final"]["mondrian"][k] for k in ("coverage", "coverage_ci95", "area_median_um2")}},
         "decision": "reported", "reasoning": "single-use final evaluation"})
print(sum(1 for _ in REG.open()), "entries")
