"""Build the corrected Phase 3/4 comparison (markdown + json) from reports/phase6/corrected_history/rerun_raw.json.
Original reports are read, never modified."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.phase5.data import load_window, window_path
from scripts.phase5.detectors import BASELINE, run_detector
from scripts.phase5.metrics import legacy_match, window_metrics

H = ROOT / "reports" / "phase6" / "corrected_history"
raw = json.loads((H / "rerun_raw.json").read_text())
orig = json.loads((ROOT / "reports" / "ibl-detection-localization.json").read_text())
pil = json.loads((ROOT / "reports" / "ibl-pilot.json").read_text())
O = {(w["pid"][:8], w["t0_s"]): w for w in orig["per_window"] + orig["second_session"]["per_window"]}
VAR = ["legacy_6sigma_no_car", "car_6sigma", "car_4sigma", "car_4sigma_hp_only", "car_4sigma_9000lp", "car_4sigma_both_polarity"]
rows, var_acc = [], {v: {"orig": [], "ctrl": [], "corr": []} for v in VAR}
for c, r in zip(raw["control"], raw["corrected"]):
    if "status" in r:
        rows.append({"window": f"{r['pid'][:8]} t0={r['t0_s']:.0f}s", "status": r["status"]}); continue
    k = (r["pid"][:8], r["t0_s"]); o = O[k]
    w = load_window(window_path(r["pid"], r["t0_s"], 1.0))
    ev = run_detector(w["voltages_uv"], BASELINE, w["coordinates_um"])["events"]
    chance = float(np.mean([legacy_match(ev, (w["sorter_times_s"] + s) % 1.0, w["fs"], 1.0)["recall"] for s in (0.041, 0.073)]))
    chance_prec = float(np.mean([legacy_match(ev, (w["sorter_times_s"] + s) % 1.0, w["fs"], 1.0)["precision"] for s in (0.041, 0.073)]))
    f = lambda x, v="car_4sigma": x["detector_variants"][v]["vs_sorting"]["1.0"]
    rows.append({"window": f"{r['pid'][:8]} t0={r['t0_s']:.0f}s", "sync_offset_ms": r["sync_offset_ms_session_minus_probe"], "duration_s": r["duration_s"],
                 "sorter_spikes": {"original": o["sorter_spikes_on_selected_contacts"], "control": c["sorter_spikes_on_selected_contacts"], "corrected": r["sorter_spikes_on_selected_contacts"]},
                 "events_car_4sigma": f(r) and r["detector_variants"]["car_4sigma"]["events"],
                 "tp_fp_fn_1ms": {"original": [f(o)[x] for x in ("true_positives", "false_positives", "false_negatives")],
                                  "control": [f(c)[x] for x in ("true_positives", "false_positives", "false_negatives")],
                                  "corrected": [f(r)[x] for x in ("true_positives", "false_positives", "false_negatives")]},
                 "recall_precision_f1_1ms": {"original": [f(o)[x] for x in ("recall", "precision", "f1")], "corrected": [f(r)[x] for x in ("recall", "precision", "f1")]},
                 "chance_recall_precision_corrected_1ms": [chance, chance_prec],
                 "legacy_6sigma_recall": {"original": f(o, "legacy_6sigma_no_car")["recall"], "corrected": f(r, "legacy_6sigma_no_car")["recall"]},
                 "sorter_aligned_fraction_above_4sigma": {"original": o["sorter_aligned_snr_improved"]["fraction_above_4_sigma"], "corrected": r["sorter_aligned_snr_improved"]["fraction_above_4_sigma"]},
                 "localization_median_axial_um": {"original": {m: v["median_axial_disagreement_um"] for m, v in o["localization_sorter_triggered"].items()},
                                                  "corrected": {m: v["median_axial_disagreement_um"] for m, v in r["localization_sorter_triggered"].items()}}})
    for v in VAR:
        for key, src in (("orig", o), ("ctrl", c), ("corr", r)):
            var_acc[v][key].append([f(src, v)[x] for x in ("recall", "precision", "f1")])
tab = {v: {k: np.mean(a, axis=0).round(3).tolist() for k, a in d.items()} for v, d in var_acc.items()}
pc = {"published": pil["median_axial_disagreement_to_sorter_peak_channel_um"], "control": raw["pilot_control"]["median_axial_disagreement_to_sorter_peak_channel_um"], "corrected": raw["pilot_corrected"]["median_axial_disagreement_to_sorter_peak_channel_um"]}
MW = json.loads((H / "missing_window_offsets_ms.json").read_text())
res = {"windows": rows, "variant_means_recall_precision_f1_1ms": tab, "pilot_localization_median_axial_um": pc,
       "pilot_fraction_above_4sigma": {"published": pil["sorter_aligned_snr_after_car"]["fraction_above_4_sigma"], "corrected": raw["pilot_corrected"]["sorter_aligned_snr_after_car"]["fraction_above_4_sigma"]},
       "decode_block_offsets": json.loads((H / "decode_block_offsets.json").read_text())}
(H / "corrected_summary.json").write_text(json.dumps(res, indent=1) + "\n")
L = ["# Corrected Phase 3/4 IBL results (probe-clock sorter alignment)", "",
     "**Status:** the original reports (`reports/ibl-*.json`, `reports/phase3-validation.md`, `reports/phase4-validation.md`) are unchanged and remain as published; their IBL detection/localization numbers are **deprecated** by this document. Produced by `scripts/phase6/corrected_history.py` + `scripts/phase6/corrected_history_report.py`.", "",
     "## What was wrong", "Phase 3-4 compared IBL `spikes.times` (session clock) with raw sample indices (probe clock) as if they were the same timebase. The two differ by a drift of roughly -10 µs per second (the offset column below), so later windows were increasingly misaligned.",
     "The *control* column re-runs the unchanged original evaluation code with the sorter times exactly as it saw them and reproduces the published numbers, which shows the clock offset accounts for the behaviour. The *corrected* column maps sorter times through `_spikeglx_*.timestamps.npy`. No detector parameter changed.", "",
     "## Frozen detector (`car_4sigma_300_6000_negative`) vs sorter, 1 ms tolerance, 1 s windows", "",
     "| Window | Offset (ms) | Sorter spikes | Events | TP/FP/FN original | TP/FP/FN control | TP/FP/FN corrected | Recall orig → corr | Precision orig → corr | Chance recall / precision (corrected) |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "status" in r:
        L.append(f"| {r['window']} | - | - | - | - | - | - | not re-runnable: {r['status']} | | |"); continue
    t = r["tp_fp_fn_1ms"]; rp = r["recall_precision_f1_1ms"]
    L.append(f"| {r['window']} | {r['sync_offset_ms']:.2f} | {r['sorter_spikes']['original']} | {r['events_car_4sigma']} | {'/'.join(map(str, t['original']))} | {'/'.join(map(str, t['control']))} | {'/'.join(map(str, t['corrected']))} | {rp['original'][0]:.3f} → {rp['corrected'][0]:.3f} | {rp['original'][1]:.3f} → {rp['corrected'][1]:.3f} | {r['chance_recall_precision_corrected_1ms'][0]:.3f} / {r['chance_recall_precision_corrected_1ms'][1]:.3f} |")
def _mean(sess, key, which):
    v = [r["legacy_6sigma_recall" if key == "legacy" else "recall_precision_f1_1ms"][which] if key == "legacy" else r["recall_precision_f1_1ms"][which][0] for r in rows if "status" not in r and r["window"].startswith(sess)]
    return float(np.mean(v))
HEAD = {sess: {"legacy_original": _mean(sess, "legacy", "original"), "legacy_corrected": _mean(sess, "legacy", "corrected"), "car4_original": _mean(sess, "car", "original"), "car4_corrected": _mean(sess, "car", "corrected")} for sess in ("4836a465", "1a60a6e1")}
res["headline_mean_recall_1ms"] = HEAD
(H / "corrected_summary.json").write_text(json.dumps(res, indent=1) + "\n")
L += ["", "**Headline (mean unconstrained 1 ms recall over the window set):** first insertion (3 windows) legacy 6σ no-CAR " + f"{HEAD['4836a465']['legacy_original']:.3f} → {HEAD['4836a465']['legacy_corrected']:.3f}, CAR 4σ {HEAD['4836a465']['car4_original']:.3f} → {HEAD['4836a465']['car4_corrected']:.3f}; second insertion (2 windows) legacy {HEAD['1a60a6e1']['legacy_original']:.3f} → {HEAD['1a60a6e1']['legacy_corrected']:.3f}, CAR 4σ {HEAD['1a60a6e1']['car4_original']:.3f} → {HEAD['1a60a6e1']['car4_corrected']:.3f}. The published CAR-4σ figures (49.0 %, 46.9 %) and the first-insertion legacy figure (3.4 %) are reproduced by the original-clock control; the published second-insertion legacy figure (8.1 %) comes from a separate legacy-detector field and is 7.8 % for the 6σ no-CAR variant tabulated here. The corrected values replace all of them.", "", "Matching is the original ordered one-to-one matcher (unconstrained, not channel-consistent), so these are directly comparable to the published figures; the chance column shows that a large share of unconstrained recall is chance at this event rate.", "",
      "## Detector-variant table (mean over the re-runnable windows, 1 ms; recall / precision / F1)", "", "| Variant | Original | Control | Corrected |", "|---|---|---|---|"]
for v, d in tab.items():
    L.append(f"| {v} | {d['orig']} | {d['ctrl']} | {d['corr']} |")
L += ["", "The ordering of variants is unchanged by the correction (CAR + 4σ >> CAR 6σ >> no-CAR 6σ; 9 kHz low-pass similar), and absolute agreement is much higher. Dual polarity has the highest F1 both before and after correction but much lower precision; the Phase 4 choice of negative-only rested on precision (false-positive inflation), not on F1, and that trade-off is unchanged. Corrected precision values of 0.9+ for the 6σ detectors reflect that they detect few, large spikes.", "",
      "## IBL localization pilot (first insertion, t0 = 100 s, 50 sorter-triggered snippets)", "", "Median axial disagreement to the sorter peak-channel depth (µm; a proxy, not localization error):", "", "| Method | Published | Control | Corrected |", "|---|---|---|---|"]
for m in pc["published"]:
    L.append(f"| {m} | {pc['published'][m]:.1f} | {pc['control'][m]:.1f} | {pc['corrected'][m]:.1f} |")
L += ["", f"Fraction of sorter-aligned troughs ≥ 4σ after CAR: {res['pilot_fraction_above_4sigma']['published']:.3f} → {res['pilot_fraction_above_4sigma']['corrected']:.3f}.",
      "The snippets were centred 1.5 ms away from the true spike peak, which degraded every method and exaggerated the apparent method differences (template worst); after correction the four methods agree within ~3 µm on this window. Conclusions that ranked localization methods on this pilot are deprecated.", "",
      "## Not corrected / unaffected", "",
      "* `ibl-decode.json` uses only session-clock quantities (sorted spikes and `stimOn_times`), so it is not affected by this error.",
      f"* `ibl-detector-decode.json` mixed clocks: detector events are on the probe clock, `stimOn_times` on the session clock; the 50-250 ms window was shifted by {res['decode_block_offsets']['A']['offset_ms_at_block_start']:.1f} ms (insertion A, t0=1465 s) and {res['decode_block_offsets']['B']['offset_ms_at_block_start']:.1f} ms (insertion B, t0=1825 s) of 200 ms. **Re-run** with trial onsets mapped to the probe clock (`scripts/phase6/corrected_decode.py`, original experiment code unchanged): within-session held-out accuracy of detector counts 0.2 → 0.4 (5 test trials; sorted counts unchanged at 0.6), cross-recording detector accuracy 0.7 → 0.3 (10 trials; majority 0.7). With 10-11 trials these changes are within noise and no decoding conclusion can be drawn either way; the original conclusion is deprecated rather than reversed.",
      *([f"* {len([r for r in rows if 'status' in r])} legacy window(s) could not be re-downloaded and are not re-run."] if any("status" in r for r in rows) else []), "",
      "## Remaining uncertainty", "Sorter spikes are not ground truth; three windows of 1 s on two insertions are a small sample; the control reproduces the published numbers to within one spike (a one-spike edge difference in one window arises from reconstructing the original clock with a constant offset). Chance level is estimated with two circular shifts (41 ms, 73 ms)."]
(H / "CORRECTED_PHASE3_4.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:20]))
