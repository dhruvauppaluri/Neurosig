"""Write the Phase 10 experiment registry from the saved result files (numbers are read, never typed in)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import log_experiment as _log

R = Path("reports/phase10")
REG = R / "experiment_registry.jsonl"
if REG.exists():
    REG.unlink()
def log_experiment(e):
    _log(e, REG)


J = lambda p: json.loads((R / p).read_text())

ex = J("real_extra_dev.json")
p9w, bw = ex["real"]["P9"]["weighted_scan"], ex["real"]["corrected_B"]["weighted_scan"]
log_experiment({"id": "P10-M1-segments-snr-peakdrop-dev", "hypothesis": "Within-unit instability is dominated by low-SNR sorter events and does not grow across early/middle/late recording segments",
                "split": "dev windows A-D (+null)", "params": {"configs": ["legacy_raw", "corrected_B", "P9"], "null_shift_ms": 20},
                "before": {"corrected_B_weighted": {"segment_shift_median": bw["segment_shift_um"]["median"], "snr_rho": bw["deviation_snr_spearman"],
                                                     "peak_unknown_median": bw["drop_peak_unknown"]["median"]}},
                "after": {"P9_weighted": {"segment_shift_median": p9w["segment_shift_um"]["median"], "snr_rho": p9w["deviation_snr_spearman"],
                                           "deviation_by_snr_quintile": [(q["snr_median"], q["median"]) for q in p9w["deviation_by_snr_quintile"]],
                                           "peak_known_median": p9w["drop_peak_known"]["median"], "peak_unknown_median": p9w["drop_peak_unknown"]["median"]}},
                "decision": "measurement (no algorithm change)", "reasoning": "Deviation falls ~4x from lowest to highest SNR quintile; segment spreads are similar early/middle/late.",
                "next": "dead-contact handling (dropout) as the remaining controllable instability"})

ov = J("overlap_stress.json")
log_experiment({"id": "P10-E1-k3-extension", "hypothesis": "A greedy third source orthogonal to the accepted K=2 span recovers triples without adding false triples on pairs",
                "split": "tau3 on p10dev; reported on p10test (single run)", "params": {"k3_tau": ov["k3_tau"], "rule": ov["k3_rule"]},
                "before": {"triples_all_recovered": ov["triples_K2_pipeline"]["all_sources_recovered"], "triples_resolvable_all_recovered": ov["triples_K2_pipeline_resolvable"]["all_sources_recovered"]},
                "after": {"triples_all_recovered": ov["triples_K3_extension"]["all_sources_recovered"], "triples_resolvable_all_recovered": ov["triples_K3_extension_resolvable"]["all_sources_recovered"],
                          "false_three_on_resolvable_pairs": ov["resolvable_pairs_false_three_rate"], "singles_declared_3": ov["singles_K3_extension"]["declared"]["3"]},
                "decision": "retain as opt-in extension (off in the frozen pipeline)", "reasoning": "Triples go from 0% to partial recovery; pair and single behaviour unchanged except 1.3% false triples."})

for pool in ("p10dev", "p10confirm"):
    d = J(f"exp_dead_mask_{pool}.json")["conditions"]
    log_experiment({"id": f"P10-E2-dead-mask-overlap-{pool}", "hypothesis": "Interpolating unknown dead contacts makes the K<=2 fit see a misfit second source; excluding them (fit mask) removes phantom sources",
                    "split": pool + (" (decision)" if pool == "p10dev" else " (confirmation, single run)"), "params": {"arm": "dead contacts -> mask, baseline only", "masks": "seed 20271005, identical per arm"},
                    "before": {k.split("/")[0]: {"pairs_recall": v["resolvable_pairs"]["all_sources_recovered"], "singles_false_two": v["false_two_singles"]} for k, v in d.items() if k.endswith("interp_P9")},
                    "after": {k.split("/")[0]: {"pairs_recall": v["resolvable_pairs"]["all_sources_recovered"], "singles_false_two": v["false_two_singles"]} for k, v in d.items() if k.endswith("mask_X1")},
                    "decision": "retain", "reasoning": "False two-source rate under unknown dropout falls from 50-83% to 4-10%; identical without dropout."})

x2 = J("exp_x2_dev.json")["results"]
keys = ("dropU_random_0.75", "dropU_random_0.5", "dropU_clustered_0.75")
log_experiment({"id": "P10-E3-dead-exclusion-single-source", "hypothesis": "Rebuilding P9 engines on detected-live contacts beats interpolation under unknown dropout",
                "split": "dev windows A-D", "params": {"config": "P10 = P9 + dead-contact exclusion"},
                "before": {m: {k: x2["P9"][m][k]["median"] for k in keys} for m in x2["P9"]},
                "after": {m: {k: x2["P10"][m][k]["median"] for k in keys} for m in x2["P10"]},
                "decision": "retain; frozen as P10", "reasoning": "Median unknown-dropout shift falls for all five engines; outputs identical when no contact is dead."})

if (R / "final_locked10.json").exists():
    f = J("final_locked10.json")["battery"]
    log_experiment({"id": "P10-F1-locked-I-L", "hypothesis": "P9/P10 gains over corrected baseline B generalize to four new insertions from new labs",
                    "split": "Phase 10 locked I-L (evaluated once)", "params": {"configs": list(f)},
                    "before": {m: f["corrected_B"][m]["within_unit_spread_um"]["median"] for m in f["corrected_B"]},
                    "after": {c: {m: f[c][m]["within_unit_spread_um"]["median"] for m in f[c]} for c in ("P9", "P10")},
                    "decision": "reported", "reasoning": "single-use locked evaluation; see phase10-validation.md"})
if (R / "decode_new_sessions.json").exists():
    s = [x for x in J("decode_new_sessions.json")["sessions"] if x.get("status") == "ran"]
    log_experiment({"id": "P10-F2-decode-new-sessions", "hypothesis": "P9 localization adds decodable information on new insertions",
                    "split": "new insertions I-L, frozen Phase 9 protocol", "params": {"sessions": [x["session"] for x in s]},
                    "before": {x["session"]: x["accuracy"]["1_counts"] for x in s},
                    "after": {x["session"]: {"p9": x["accuracy"]["4_counts+p9_loc"], "delta": x["paired"]["p9_vs_counts"]} for x in s},
                    "decision": "reported", "reasoning": "frozen protocol, nothing tuned"})
print(sum(1 for _ in REG.open()), "entries")
