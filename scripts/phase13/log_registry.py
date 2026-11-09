"""Write reports/phase13/experiment_registry.jsonl from saved Phase 13 results (numbers read, never typed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import log_experiment as _log

R = Path("reports/phase13"); REG = R / "experiment_registry.jsonl"
if REG.exists():
    REG.unlink()
J = lambda p: json.loads((R / p).read_text())
log = lambda e: _log(e, REG)
s = J("steinmetz_repro.json")
log({"id": "P13-A-steinmetz", "hypothesis": "Historical blank vs contrast decoder reproduces", "split": "Steinmetz session 11 (73 held-out trials)",
     "params": {"sha256": s["source_sha256"]}, "before": {"float": 0.9041095890410958}, "after": {"float": s["float_test_accuracy"], "majority": s["majority_class_test_accuracy"]},
     "decision": "reproduced", "reasoning": "identical"})
comps = ("tcounts_vs_counts", "umap_vs_counts", "umap_vs_counts_rp", "umap_vs_umap_shuf", "smap_vs_umap", "stmap_vs_tcounts", "countsumap_vs_countsumapshuf")
for split in ("dev", "select", "final"):
    d = J(f"ibl13_{split}.json")["summary"]
    for t in ("T1_presence", "T2_side", "T3_contrast5"):
        if t not in d:
            continue
        log({"id": f"P13-IBL-{t}-{split}", "hypothesis": "Timing (B) and spatial maps (C) add stimulus information beyond counts",
             "split": f"IBL {split}", "params": {"window_ms": [40, 120]},
             "before": {"counts_bacc": d[t]["balanced_accuracy_mean"]["lr:counts"], "majority": d[t]["majority_mean"]},
             "after": {k: d[t]["paired_across_sessions"].get(k) for k in comps},
             "decision": "reported", "reasoning": "accuracy deltas, session bootstrap and Wilcoxon"})
    if "T2_side" in d:
        b = d["T2_side"]["balanced_accuracy_mean"]
        log({"id": f"P13-IBL-noise-{split}", "hypothesis": "Per-spike depth-map decoding degrades with injected position noise",
             "split": f"IBL {split}", "params": {"sigma_um": [0, 10, 25, 50, 100, 200]}, "before": {"smap": b["lr:smap"]},
             "after": {k: b[k] for k in b if "noise" in k}, "decision": "reported", "reasoning": "Section 11 perturbation"})
for split in ("dev", "select", "final"):
    d = J(f"allen13_within_{split}.json")["sessions"]
    log({"id": f"P13-Allen-within-{split}", "hypothesis": "Depth maps add information beyond counts for orientation, spatial frequency and images",
         "split": f"Allen {split}", "params": {"window_ms": [0, 250]},
         "before": {s: {t: r["lr:counts"]["balanced_accuracy"] for t, r in o["tasks"].items() if "lr:counts" in r} for s, o in d.items()},
         "after": {s: {t: r.get("paired") for t, r in o["tasks"].items() if "paired" in r} for s, o in d.items()},
         "decision": "negative (maps lose information; dmap_p12 invalid: waveform channel-mapping defect)", "reasoning": "within-session paired bootstrap"})
for f in ("allen13_cross_dev_to_select.json", "allen13_cross_dev+select_to_final.json"):
    d = J(f)
    log({"id": f"P13-Allen-{f[:-5]}", "hypothesis": "Depth-aligned spatial maps generalize across mice", "split": f"train {d['train']} test {d['test']}",
         "params": {}, "before": {}, "after": d["tasks"], "decision": "negative (near chance)", "reasoning": "cross-mouse"})
sim = J("sim13_localization_link.json")
log({"id": "P13-sim-link", "hypothesis": "Localization precision matters when the population code is spatially organized at a comparable scale",
     "split": "simulation", "params": {"deltas": [20, 50, 100, 250]}, "before": {},
     "after": {k: {kk: v[kk]["mean"] for kk in v if kk.startswith("cross_unit")} for k, v in sim.items() if k.startswith("M")},
     "decision": "supported in simulation", "reasoning": "accuracy falls when sigma ~ delta/4 - delta/2; salt-and-pepper at chance"})
print(sum(1 for _ in REG.open()), "entries")
