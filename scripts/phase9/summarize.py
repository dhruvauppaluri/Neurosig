import json, numpy as np
R = json.load(open("reports/phase9/final_locked.json"))
M = ["amplitude", "template", "beamform", "weighted_scan", "robust_scan"]
out = {"locked": {}, "heldout5_previously_used": {}}
for kind, key in (("locked", "locked"), ("heldout5", "heldout5_previously_used")):
    res = R["sets"][kind]["results"]
    B, P = res["corrected_B"], res["P9"]
    d = {}
    for m in M:
        row = {}
        row["axial_median_change_pct"] = 100 * (P[m]["axial_disagreement_um"]["median"] / B[m]["axial_disagreement_um"]["median"] - 1)
        row["spread_change_pct"] = 100 * (P[m]["within_unit_spread_um"]["median"] / B[m]["within_unit_spread_um"]["median"] - 1)
        row["spread_p90_change_pct"] = 100 * (P[m]["within_unit_spread_um"]["p90"] / B[m]["within_unit_spread_um"]["p90"] - 1)
        row["valid_fraction"] = [B[m]["valid_fraction"], P[m]["valid_fraction"]]
        row["runtime_ratio"] = P[m]["runtime_ms_median"] / B[m]["runtime_ms_median"]
        for grp, keys in (("dropout_known", [k for k in B[m] if k.startswith("drop_")]), ("dropout_unknown", [k for k in B[m] if k.startswith("dropU_")]),
                          ("noise", [k for k in B[m] if k.startswith("noise_")])):
            ch = [100 * (P[m][k]["median"] / max(B[m][k]["median"], 1e-9) - 1) for k in keys if P[m][k].get("n") and B[m][k].get("n") and B[m][k]["median"] > 0.5]
            row[grp + "_median_change_pct"] = float(np.mean(ch))
            row[grp + "_fraction_improved"] = float(np.mean([c < 0 for c in ch]))
        row["noise_p90_change_pct"] = float(np.mean([100 * (P[m][k]["p90"] / max(B[m][k]["p90"], 1e-9) - 1) for k in B[m] if k.startswith("noise_") and P[m][k].get("n")]))
        d[m] = row
    bestB = min(B[m]["within_unit_spread_um"]["median"] for m in M); bestP = min(P[m]["within_unit_spread_um"]["median"] for m in M)
    d["best_vs_best_spread_change_pct"] = 100 * (bestP / bestB - 1)
    d["per_session_spread_P9_vs_B_pct"] = {m: {s: 100 * (P[m]["per_session_median_spread_um"][s] / B[m]["per_session_median_spread_um"][s] - 1) for s in P[m]["per_session_median_spread_um"]} for m in M}
    out[key] = d
D = json.load(open("reports/phase9/decode_ablation.json"))["sessions"]
ok = [s for s in D if s["status"] == "ran"]
dec = {"n_sessions": len(ok), "mean_accuracy_pct": {k: float(100 * np.mean([s["accuracy"][k] for s in ok])) for k in ok[0]["accuracy"]},
       "majority_mean_pct": float(100 * np.mean([s["majority_accuracy"] for s in ok]))}
rng = np.random.default_rng(0)
for k in ok[0]["paired"]:
    v = 100 * np.array([s["paired"][k]["delta"] for s in ok]); bs = [rng.choice(v, len(v)).mean() for _ in range(5000)]
    dec[k] = {"mean_pts": float(v.mean()), "ci95_sessions": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_positive": int((v > 0).sum()), "per_session": [round(float(x), 2) for x in v]}
dec["spatial_features_explained_by_counts_R2_median_of_sessions"] = float(np.median([s["spatial_features_explained_by_counts_oof_R2"]["median"] for s in ok]))
dec["per_session_p9_minus_counts_pts_with_region"] = {s["session"]: [s["region_prefix"], round(100 * s["paired"]["p9_vs_counts"]["delta"], 2), round(100 * (s["accuracy"]["1_counts"] - s["majority_accuracy"]), 1)] for s in ok}
out["decode"] = dec
json.dump(out, open("reports/phase9/summary.json", "w"), indent=1, default=float)
for kind in ("locked", "heldout5_previously_used"):
    print("=====", kind, "best-vs-best spread change %.1f%%" % out[kind]["best_vs_best_spread_change_pct"])
    for m in M:
        r = out[kind][m]
        print(f"{m:13s} axial {r['axial_median_change_pct']:+5.0f}% spread {r['spread_change_pct']:+5.0f}% p90 {r['spread_p90_change_pct']:+5.0f}% valid {r['valid_fraction'][0]:.2f}->{r['valid_fraction'][1]:.2f} rt x{r['runtime_ratio']:.1f} | drop known {r['dropout_known_median_change_pct']:+5.0f}% unk {r['dropout_unknown_median_change_pct']:+5.0f}% noise {r['noise_median_change_pct']:+5.0f}% (p90 {r['noise_p90_change_pct']:+5.0f}%, improved {100*r['noise_fraction_improved']:.0f}% of conds)")
print(json.dumps({k: v for k, v in dec.items() if k != "per_session_p9_minus_counts_pts_with_region"}, indent=None)[:2500])
print(dec["per_session_p9_minus_counts_pts_with_region"])
