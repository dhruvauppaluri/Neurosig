# Corrected Phase 3/4 IBL results (probe-clock sorter alignment)

**Status:** the original reports (`reports/ibl-*.json`, `reports/phase3-validation.md`, `reports/phase4-validation.md`) are unchanged and remain as published; their IBL detection/localization numbers are **deprecated** by this document. Produced by `scripts/phase6/corrected_history.py` + `scripts/phase6/corrected_history_report.py`.

## What was wrong
Phase 3-4 compared IBL `spikes.times` (session clock) with raw sample indices (probe clock) as if they were the same timebase. The two differ by a drift of roughly -10 µs per second (the offset column below), so later windows were increasingly misaligned.
The *control* column re-runs the unchanged original evaluation code with the sorter times exactly as it saw them and reproduces the published numbers, which shows the clock offset accounts for the behaviour. The *corrected* column maps sorter times through `_spikeglx_*.timestamps.npy`. No detector parameter changed.

## Frozen detector (`car_4sigma_300_6000_negative`) vs sorter, 1 ms tolerance, 1 s windows

| Window | Offset (ms) | Sorter spikes | Events | TP/FP/FN original | TP/FP/FN control | TP/FP/FN corrected | Recall orig → corr | Precision orig → corr | Chance recall / precision (corrected) |
|---|---|---|---|---|---|---|---|---|---|
| 4836a465 t0=50s | -0.73 | 205 | 242 | 157/85/48 | 157/85/48 | 160/82/45 | 0.766 → 0.780 | 0.649 → 0.661 | 0.351 / 0.298 |
| 4836a465 t0=100s | -1.47 | 294 | 236 | 110/126/184 | 110/126/184 | 185/51/109 | 0.374 → 0.629 | 0.466 → 0.784 | 0.303 / 0.377 |
| 4836a465 t0=200s | -2.90 | 292 | 240 | 96/144/196 | 96/144/196 | 186/54/106 | 0.329 → 0.637 | 0.400 → 0.775 | 0.281 / 0.342 |
| 1a60a6e1 t0=50s | -0.50 | 342 | 269 | 206/63/136 | 207/62/136 | 212/57/131 | 0.602 → 0.618 | 0.766 → 0.788 | 0.399 / 0.509 |
| 1a60a6e1 t0=90s | -0.87 | 904 | 324 | 304/20/600 | 306/18/598 | 310/14/594 | 0.336 → 0.343 | 0.938 → 0.957 | 0.264 / 0.736 |

**Headline (mean unconstrained 1 ms recall over the window set):** first insertion (3 windows) legacy 6σ no-CAR 0.034 → 0.039, CAR 4σ 0.490 → 0.682; second insertion (2 windows) legacy 0.078 → 0.078, CAR 4σ 0.469 → 0.480. The published CAR-4σ figures (49.0 %, 46.9 %) and the first-insertion legacy figure (3.4 %) are reproduced by the original-clock control; the published second-insertion legacy figure (8.1 %) comes from a separate legacy-detector field and is 7.8 % for the 6σ no-CAR variant tabulated here. The corrected values replace all of them.

Matching is the original ordered one-to-one matcher (unconstrained, not channel-consistent), so these are directly comparable to the published figures; the chance column shows that a large share of unconstrained recall is chance at this event rate.

## Detector-variant table (mean over the re-runnable windows, 1 ms; recall / precision / F1)

| Variant | Original | Control | Corrected |
|---|---|---|---|
| legacy_6sigma_no_car | [0.052, 0.639, 0.094] | [0.051, 0.639, 0.094] | [0.054, 0.897, 0.099] |
| car_6sigma | [0.169, 0.757, 0.268] | [0.169, 0.764, 0.27] | [0.226, 0.984, 0.36] |
| car_4sigma | [0.481, 0.644, 0.53] | [0.482, 0.646, 0.531] | [0.602, 0.793, 0.662] |
| car_4sigma_hp_only | [0.46, 0.652, 0.522] | [0.461, 0.654, 0.523] | [0.566, 0.798, 0.644] |
| car_4sigma_9000lp | [0.473, 0.652, 0.529] | [0.474, 0.655, 0.53] | [0.591, 0.809, 0.663] |
| car_4sigma_both_polarity | [0.565, 0.557, 0.533] | [0.566, 0.559, 0.534] | [0.737, 0.7, 0.687] |

The ordering of variants is unchanged by the correction (CAR + 4σ >> CAR 6σ >> no-CAR 6σ; 9 kHz low-pass similar), and absolute agreement is much higher. Dual polarity has the highest F1 both before and after correction but much lower precision; the Phase 4 choice of negative-only rested on precision (false-positive inflation), not on F1, and that trade-off is unchanged. Corrected precision values of 0.9+ for the 6σ detectors reflect that they detect few, large spikes.

## IBL localization pilot (first insertion, t0 = 100 s, 50 sorter-triggered snippets)

Median axial disagreement to the sorter peak-channel depth (µm; a proxy, not localization error):

| Method | Published | Control | Corrected |
|---|---|---|---|
| amplitude | 40.9 | 40.6 | 21.7 |
| template | 80.0 | 90.0 | 20.0 |
| beamform | 61.9 | 68.5 | 19.4 |
| robust_scan | 42.2 | 48.5 | 22.6 |

Fraction of sorter-aligned troughs ≥ 4σ after CAR: 0.320 → 0.707.
The snippets were centred 1.5 ms away from the true spike peak, which degraded every method and exaggerated the apparent method differences (template worst); after correction the four methods agree within ~3 µm on this window. Conclusions that ranked localization methods on this pilot are deprecated.

## Not corrected / unaffected

* `ibl-decode.json` uses only session-clock quantities (sorted spikes and `stimOn_times`), so it is not affected by this error.
* `ibl-detector-decode.json` mixed clocks: detector events are on the probe clock, `stimOn_times` on the session clock; the 50-250 ms window was shifted by -21.0 ms (insertion A, t0=1465 s) and -17.3 ms (insertion B, t0=1825 s) of 200 ms. **Re-run** with trial onsets mapped to the probe clock (`scripts/phase6/corrected_decode.py`, original experiment code unchanged): within-session held-out accuracy of detector counts 0.2 → 0.4 (5 test trials; sorted counts unchanged at 0.6), cross-recording detector accuracy 0.7 → 0.3 (10 trials; majority 0.7). With 10-11 trials these changes are within noise and no decoding conclusion can be drawn either way; the original conclusion is deprecated rather than reversed.

## Remaining uncertainty
Sorter spikes are not ground truth; three windows of 1 s on two insertions are a small sample; the control reproduces the published numbers to within one spike (a one-spike edge difference in one window arises from reconstructing the original clock with a constant offset). Chance level is estimated with two circular shifts (41 ms, 73 ms).
