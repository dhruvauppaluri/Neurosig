"""Phase 11C: difficult-session diagnostics on the 12 previously used sessions (A-L) from cached 11A selection/dev
records, raw windows, footprint events and IBL sorter metrics. Writes reports/phase11/exp11c_sessions.json.

Failure categories (thresholds fixed before inspecting F/K in this phase, from the pooled distribution of all 12):
  unreliable_reference  sorter-event trough fraction < 0.40
  low_snr               median sorter-event SNR < 3.5
  poor_isolation        fraction of window units labelled 'good' by IBL < 0.25
  noisy_or_dead         > 2 dead/flat contacts or median contact MAD > 1.5x the pooled median
  overlap_dense         > 50% of sorter events with another sorter spike within 1 ms and 100 um
Interventions per category are read from the 11A (B detector / C alignment), 11B (linefree) and 11D results.
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.events import window_array
from edge.phase9.improved import dead_channels
from scripts.phase5.data import load_window, window_path
from scripts.real_detector import frozen_real_detector, preprocess

sel = pickle.load(open("data/phase11/exp11a_select_K6.pkl", "rb"))
dev = pickle.load(open("data/phase11/exp11a_dev_K6.pkl", "rb"))
metrics = pickle.load(open("data/phase11/cluster_metrics_plain.pkl", "rb"))   # plain-list copy (no pandas in .venv)
fp = pickle.load(open("data/phase11/footprint_select_all.pkl", "rb")) + pickle.load(open("data/phase11/footprint_dev.pkl", "rb"))
sel11a = json.load(open("reports/phase11/exp11a_select_K6.json"))["per_session"]
dev11a = json.load(open("reports/phase11/exp11a_dev_K6.json"))["per_session"]
b11 = json.load(open("reports/phase11/exp11b_select.json"))["models"]

wins = H.manifest_windows("dev") + H.manifest_windows("heldout5") + H.manifest_windows("locked") + H.manifest_windows("locked10")
S = defaultdict(lambda: defaultdict(list))
for w in wins:
    p = window_path(w["pid"], w["t0_s"], w["duration_s"])
    if not p.exists():
        continue
    d = load_window(p); s = w["session"]
    f, _ = preprocess(d["voltages_uv"], frozen_real_detector(d["fs"]))
    mad = np.median(np.abs(f - np.median(f, axis=0)), axis=0) / 0.6745
    S[s]["mad_uv"].append(float(np.median(mad)))
    S[s]["n_dead"].append(int(dead_channels(d["voltages_uv"] - np.median(d["voltages_uv"], axis=0), 30).sum()))
    E = window_array(d).xyz_um(); S[s]["pitch_y"] = float(np.median(np.diff(np.unique(E[:, 1])))); S[s]["x_cols"] = int(len(np.unique(E[:, 0])))
    st = np.rint(d["sorter_times_s"] * d["fs"]).astype(int); ch = np.asarray(d["sorter_local_channel"], int)
    for i in range(len(st)):
        near = (np.abs(st - st[i]) <= int(0.001 * d["fs"])); near[i] = False
        S[s]["overlap"].append(bool(np.any(near & (np.linalg.norm(E[ch] - E[ch[i]], axis=1) <= 100))))
    for cl in np.unique(d["sorter_clusters"]):
        S[s]["units"].append(int(cl))
pooled_mad = np.median([np.median(S[s]["mad_uv"]) for s in S])
out = {"sessions": {}, "categories_rule": __doc__}
for s in sorted(S):
    src = sel11a.get(s) or dev11a.get(s)
    fsA = src["full_stream"]["A"]; m = src["matched"]["engines"]["weighted_scan"]
    mt = metrics.get(s)
    units = set(S[s]["units"])
    good = amp = cont = None
    if mt is not None:
        idx = [i for i, c in enumerate(mt["cluster_id"]) if int(c) in units]
        if idx:
            good = float(np.mean([mt["label"][i] >= 1 for i in idx]))
            amp = float(np.nanmedian([mt["amp_median"][i] for i in idx]) * 1e6)
            cont = float(np.nanmedian([mt["contamination"][i] for i in idx]))
    ev = [e for e in fp if e["session"] == s]
    widths = [float(np.sum(-e["wave"][:, int(np.argmax(e["a"]))] > e["a"].max() / 2) / 30.0) for e in ev if e["a"].max() > 0]
    extent = [int(np.sum(e["a"] > 0.3 * e["a"].max())) for e in ev if e["a"].max() > 0]
    recsS = [r for r in (sel["recs"] + dev["recs"]) if r["session"] == s and r["strategy"] == "A" and not r["null"]]
    snr = np.array([r["snr"] for r in recsS])
    ws = [w for w in sel["wstats"] + dev["wstats"] if w.get("session") == s]
    cs = np.concatenate([w["c_shift_samples"] for w in ws]) if ws else np.zeros(0)
    row = {"trough_fraction": fsA["trough_fraction"], "snr_median": fsA["snr_median"],
           "snr_p25_p75": [float(np.percentile(snr, 25)), float(np.percentile(snr, 75))],
           "matched_frac_of_sorter": src["events"]["matched_frac_of_sorter"], "chance_frac": src["events"]["chance_frac_of_sorter"],
           "contact_mad_uv_median": float(np.median(S[s]["mad_uv"])), "dead_contacts_max": int(np.max(S[s]["n_dead"])),
           "pitch_y_um": S[s]["pitch_y"], "x_columns": S[s]["x_cols"],
           "overlap_fraction": float(np.mean(S[s]["overlap"])), "ibl_good_fraction": good, "ibl_amp_median_uv": amp,
           "ibl_contamination_median": cont, "width_ms_median": float(np.median(widths)) if widths else None,
           "extent_contacts_median": float(np.median(extent)) if extent else None,
           "alignment_at_bound_fraction": float(np.mean(np.abs(cs) >= 6)) if cs.size else None,
           "spread_A_full_um": fsA["engines"]["weighted_scan"]["spread_median"],
           "spread_matched_um": {k: m[k]["spread_median"] for k in "ABC"}, "p90_matched_um": {k: m[k]["dev_p90"] for k in "ABC"},
           "spread_linefree_vs_line20_um": [b11["line20"]["per_session_spread_um"].get(s), b11["linefree"]["per_session_spread_um"].get(s)]}
    cat = []
    if row["trough_fraction"] < 0.40: cat.append("unreliable_reference")
    if row["snr_median"] < 3.5: cat.append("low_snr")
    if good is not None and good < 0.25: cat.append("poor_isolation")
    if row["dead_contacts_max"] > 2 or row["contact_mad_uv_median"] > 1.5 * pooled_mad: cat.append("noisy_or_dead")
    if row["overlap_fraction"] > 0.5: cat.append("overlap_dense")
    row["categories"] = cat
    out["sessions"][s] = row
    print(s, cat, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k in
                   ("trough_fraction", "snr_median", "matched_frac_of_sorter", "contact_mad_uv_median", "overlap_fraction",
                    "ibl_good_fraction", "ibl_amp_median_uv", "width_ms_median", "extent_contacts_median", "spread_A_full_um", "dead_contacts_max")})
# correlation of session difficulty with candidate causes (12 sessions; descriptive only)
from scipy.stats import spearmanr
keys = ["trough_fraction", "snr_median", "matched_frac_of_sorter", "contact_mad_uv_median", "overlap_fraction", "ibl_good_fraction",
        "ibl_amp_median_uv", "width_ms_median", "extent_contacts_median"]
y = [out["sessions"][s]["spread_A_full_um"] for s in out["sessions"]]
out["spearman_vs_full_spread_n12"] = {k: float(spearmanr([out["sessions"][s][k] for s in out["sessions"]], y).statistic) for k in keys
                                      if all(out["sessions"][s][k] is not None for s in out["sessions"])}
print(json.dumps(out["spearman_vs_full_spread_n12"]))
Path("reports/phase11/exp11c_sessions.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
