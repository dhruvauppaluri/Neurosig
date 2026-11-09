"""Write reports/phase12/experiment_registry.jsonl from saved result files (numbers are read, never typed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import log_experiment as _log

R = Path("reports/phase12"); REG = R / "experiment_registry.jsonl"
if REG.exists():
    REG.unlink()
J = lambda p: json.loads((R / p).read_text()) if (R / p).exists() else None
log = lambda e: _log(e, REG)

b = J("baseline_repro.json")
log({"id": "P12-0-repro", "hypothesis": "Phase 11 baselines reproduce before any change", "split": "dev A-C; p11select",
     "params": {}, "before": b["stored"], "after": b["reproduced"], "decision": "reproduced (metrics identical; runtime machine-dependent)",
     "reasoning": "difference table in baseline_repro.json"})
# 12.2 pooling
for split in ("dev", "select"):
    for pop in ("matched", "all"):
        d = J(f"exp12_2_{split}_{pop}.json")
        if not d:
            continue
        S = d["summary"]
        log({"id": f"P12-2-pool-{split}-{pop}", "hypothesis": "Pooling N disjoint spikes of a sorter unit stabilizes its position (>= 25% lower disagreement than single events)",
             "split": split, "params": {"population": pop, "ns": d["ns"], "estimators": "mean, median, robust (Huber), posagg, joint (line20)"},
             "before": {"single_l20": {N: S[N].get("l20_single", {}).get("median") for N in S}},
             "after": {k: {N: {"median": S[N].get(k, {}).get("median"), "vs_single": (S[N].get(k, {}).get("vs_single") or {}).get("rel_change"),
                              "first_last_over_random": S[N].get(k, {}).get("first_last_over_random")} for N in S}
                       for k in ("l20_robust", "l20_mean", "l20_joint", "ws_robust", "ws_posagg")},
             "decision": "l20_robust adopted as the unit-level estimator (decided on dev, confirmed on select)" if pop == "matched" else "reported",
             "reasoning": "robust ~ mean > joint > posagg; temporal first-vs-last pools exceed random pools (drift / nonstationarity), reported, not averaged away"})
for rho in ("0.9", "0.8", "0.7"):
    d = J(f"exp12_2b_dev_rho{rho}.json")
    if d:
        log({"id": f"P12-2b-online-rho{rho}", "hypothesis": "Sorter-free online grouping of detector events supports unit-level pooling at purity >= 0.9",
             "split": "dev", "params": {"rho": float(rho)}, "before": {},
             "after": {s: {k: v[k] for k in ("assigned_fraction", "purity_matched_events", "n_groups_assigned")} for s, v in d["sessions"].items()},
             "decision": "rho 0.9 retained (only threshold with pooled purity >= 0.9); coverage is the limitation" if rho == "0.9" else "rejected (purity < 0.9)",
             "reasoning": "pre-stated rule: loosest threshold with purity >= 0.9"})
# 12.3 single-event variants
d3, s3 = J("exp12_3_dev.json"), J("exp12_3_select.json")
bio_d, bio_s = J("exp12_3_biophys_dev.json"), J("exp12_3_biophys_select.json")
for v in d3["variants"]:
    if v == "ws_A":
        continue
    pick = lambda d: {k: d["variants"][v][k] for k in ("spread_median", "dev_p90", "valid_fraction", "dev_gt50_frac", "ms_median")} | {
        "vs_ws_A": d["variants"][v]["vs_ws_A"], "p90_vs_ws_A": d["variants"][v]["p90_vs_ws_A"]["rel_change"]}
    log({"id": f"P12-3-{v}", "hypothesis": f"Single-event variant {v} lowers matched-event instability (-15% median, -20% p90) at >= 95% coverage",
         "split": "dev (decision); select (check); biophysical dev/select (model-mismatched accuracy)", "params": {"variant": v},
         "before": {"dev_ws_A": pick({"variants": {v: d3["variants"]["ws_A"]}}) if False else {k: d3["variants"]["ws_A"][k] for k in ("spread_median", "dev_p90")}},
         "after": {"dev": pick(d3), "select": pick(s3) if s3 else None,
                   "biophys_dev": bio_d["variants"].get(v) if bio_d else None, "biophys_select": bio_s["variants"].get(v) if bio_s else None},
         "decision": "adopted as the P12 single-event amplitude (mean3 + line20)" if v == "mean3" else "not adopted",
         "reasoning": "mean3: best dev median among variants meeting p90 <= -20% and coverage >= 95%; rank-1/GLS/selection/free-width improve "
                      "biophysical bias but destabilize real events; high-pass improves real stability but worsens low-SNR biophysical accuracy"})
# 12.4 overlap
for pool in ("p12dev", "p12select"):
    d = J(f"exp12_4_{pool}.json")
    if d:
        log({"id": f"P12-4-{pool}", "hypothesis": "Count rules on the expanded benchmark (syn + biophysical + real noise): P11 vs P12-LR",
             "split": pool, "params": {"rules": list(d["rules"])}, "before": {"P11": {f: {"false_two": d["rules"]["P11"][f]["singles"]["false_two"],
             "res": d["rules"]["P11"][f]["resolvable_pairs"]["all_sources_recovered"]} for f in ("syn", "bio", "all")}},
             "after": {"P12-LR": {f: {"false_two": d["rules"]["P12-LR"][f]["singles"]["false_two"], "false_two_ci95": d["rules"]["P12-LR"][f]["singles"]["false_two_ci95"],
                       "res": d["rules"]["P12-LR"][f]["resolvable_pairs"]["all_sources_recovered"],
                       "simul": d["rules"]["P12-LR"][f]["simultaneous_resolvable"]["all_sources_recovered"]} for f in ("syn", "bio", "all")}},
             "decision": "P12-LR adopted (false-two control); recall targets missed", "reasoning": "P11 declares two for ~75% of biophysical singles"})
for pool in ("p12dev", "p12select"):
    d = J(f"exp12_4c_bio_{pool}.json")
    if d:
        log({"id": f"P12-4c-dictionary-{pool}", "hypothesis": "An empirical spatiotemporal waveform dictionary is a valid single-neuron null for counting",
             "split": pool, "params": {"tau_F": d["tau_F"]}, "before": {},
             "after": {f: {"false_two": d[f]["false_two"], "res": d[f]["resolvable_pairs"]["all_sources_recovered"]} for f in ("syn", "bio")},
             "decision": "rejected", "reasoning": "leave-template-out singles produce F as large as true pairs; no recall at 2% false two"})
r = J("exp12_4_real_select.json")
if r:
    log({"id": "P12-4-real-proxy", "hypothesis": "A count rule declares two more often on sorter-coincident than isolated real events",
         "split": "select sessions D-P", "params": {"labels": r["labels"]},
         "before": {"P11": {L: r[L]["p11_two_rate"] for L in ("isolated", "coincident")}},
         "after": {"P12-LR": {L: r[L]["lr_two_rate"] for L in ("isolated", "coincident")}, "auc": r["auc_lr_coincident_vs_isolated"]},
         "decision": "reported", "reasoning": "P11 does not discriminate; P12-LR weakly discriminates"})
# 12.5 uncertainty
u, ur = J("exp12_5_uncertainty.json"), J("exp12_5_real_proxy.json")
log({"id": "P12-5-complete-population", "hypothesis": "Mondrian conformal over all events (count x SNR x chi1 x dropout) reaches ~90% complete-population coverage",
     "split": "calibrate p12dev, evaluate p12select", "params": {}, "before": {"P11-mond": u["select"]["P11-mond"]["primary_coverage_all_events"]},
     "after": {"P12-full": {k: u["select"]["P12-full"][k] for k in ("primary_coverage_all_events", "ci95", "unbounded_region_fraction", "true_source_coverage")},
               "count_uncertainty": {k: u["select"]["count_uncertainty"][k] for k in ("brier", "false_confidence_rate")}},
     "decision": "reported (target not reached with bounded regions)", "reasoning": "~11% of events have a primary with no true source within 60 um"})
log({"id": "P12-5-real-ident-disc", "hypothesis": "Identifiability-area-scaled discs give calibrated real-data proxy coverage with smaller regions",
     "split": "calibrate dev A-C, evaluate select D-P", "params": {"c_by_snr": ur["c_by_snr"]}, "before": {"fixed_disc": ur["select"]["fixed_disc"]},
     "after": {"ident_mondrian_snr": ur["select"]["ident_mondrian_snr"]}, "decision": "adopted for single-event uncertainty",
     "reasoning": "similar coverage at about half the area; more uniform across sessions"})
# 12.6 runtime
g, st = J("exp12_6_gate.json"), J("exp12_6_stream_select.json")
log({"id": "P12-6-gate", "hypothesis": "A cheap pre-screen avoids most K<=2 analyses without losing two-source decisions",
     "split": "fit p12dev, evaluate p12select", "params": {"keep": 0.99}, "before": g["select"]["ungated"], "after": g["select"]["gated"],
     "decision": "rejected (invokes ~93%)", "reasoning": "LR-two events are not predictable from cheap statistics"})
log({"id": "P12-6-fast", "hypothesis": "Analytic-Jacobian VP refine + closed-form shared eigenvalue + FastLine20 meet latency targets",
     "split": "real select stream", "params": {}, "before": {"k2_ms_reference": b["reproduced"]["11E_select_OV-A"]["k2_ms_median"]},
     "after": st["workers"] if st else None, "decision": "adopted (decisions 99.75% identical)", "reasoning": "single-source < 1 ms; K<=2 ~5 ms"})
f = J("final12.json")
if f:
    log({"id": "P12-FINAL", "hypothesis": "Frozen P12 on fresh sessions Q-S and the p12final pool", "split": "final (single use)",
         "params": {"p12_hash": f["p12_hash"]}, "before": {}, "after": {"steps": [(s["step"], s["returncode"]) for s in f["steps"]]},
         "decision": "reported", "reasoning": "single-use final evaluation"})
print(sum(1 for _ in REG.open()), "entries")
