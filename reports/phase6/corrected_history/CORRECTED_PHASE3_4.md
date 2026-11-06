# Corrected Phase 3/4 IBL results (probe-clock sorter alignment)

**Status:** the original reports (`reports/ibl-*.json`, `reports/phase3-validation.md`, `reports/phase4-validation.md`) are unchanged and remain as published; their IBL detection/localization numbers are **deprecated** by this document. Produced by `scripts/phase6/corrected_history.py` + `scripts/phase6/corrected_history_report.py`.

## What was wrong
Phase 3-4 compared IBL `spikes.times` (session clock) with raw sample indices (probe clock) as if they were the same timebase. The two differ by a drift of roughly -10 µs per second (the offset column below), so later windows were increasingly misaligned.
The *control* column re-runs the unchanged original evaluation code with the sorter times exactly as it saw them and reproduces the published numbers, which shows the clock offset accounts for the behaviour. The *corrected* column maps sorter times through `_spikeglx_*.timestamps.npy`. No detector parameter changed.

## Frozen detector (`car_4sigma_300_6000_negative`) vs sorter, 1 ms tolerance, 1 s windows

| Window | Offset (ms) | Sorter spikes | Events | TP/FP/FN original | TP/FP/FN control | TP/FP/FN corrected | Recall orig → corr | Precision orig → corr | Chance recall / precision (corrected) |
|---|---|---|---|---|---|---|---|---|---|
| 4836a465 t0=50s | - | - | - | - | - | - | not re-runnable: window not available (download refused: HTTP 403) | | |
| 4836a465 t0=100s | -1.47 | 294 | 236 | 110/126/184 | 110/126/184 | 185/51/109 | 0.374 → 0.629 | 0.466 → 0.784 | 0.303 / 0.377 |
| 4836a465 t0=200s | -2.90 | 292 | 240 | 96/144/196 | 96/144/196 | 186/54/106 | 0.329 → 0.637 | 0.400 → 0.775 | 0.281 / 0.342 |
| 1a60a6e1 t0=50s | -0.50 | 342 | 269 | 206/63/136 | 207/62/136 | 212/57/131 | 0.602 → 0.618 | 0.766 → 0.788 | 0.399 / 0.509 |
| 1a60a6e1 t0=90s | - | - | - | - | - | - | not re-runnable: window not available (download refused: HTTP 403) | | |

Matching is the original ordered one-to-one matcher (unconstrained, not channel-consistent), so these are directly comparable to the published figures; the chance column shows that a large share of unconstrained recall is chance at this event rate.

## Detector-variant table (mean over the re-runnable windows, 1 ms; recall / precision / F1)

| Variant | Original | Control | Corrected |
|---|---|---|---|
| legacy_6sigma_no_car | [0.043, 0.492, 0.077] | [0.042, 0.492, 0.077] | [0.046, 0.908, 0.083] |
| car_6sigma | [0.132, 0.615, 0.217] | [0.133, 0.62, 0.218] | [0.228, 0.986, 0.368] |
| car_4sigma | [0.435, 0.544, 0.483] | [0.435, 0.545, 0.484] | [0.628, 0.782, 0.697] |
| car_4sigma_hp_only | [0.413, 0.557, 0.474] | [0.413, 0.558, 0.475] | [0.58, 0.786, 0.667] |
| car_4sigma_9000lp | [0.425, 0.557, 0.482] | [0.426, 0.558, 0.483] | [0.611, 0.799, 0.693] |
| car_4sigma_both_polarity | [0.543, 0.48, 0.509] | [0.543, 0.481, 0.51] | [0.775, 0.675, 0.721] |

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
* `ibl-detector-decode.json` mixed clocks: detector events are on the probe clock, `stimOn_times` on the session clock; the 50-250 ms window was shifted by -21.0 ms (insertion A, t0=1465 s) and -17.3 ms (insertion B, t0=1825 s) of 200 ms. It was **not re-run** (the 40 s raw blocks could not be downloaded at the time of this phase; the data server returned HTTP 403). The experiment had 10-11 trials and was already inconclusive; treat it as deprecated.
* Two of the five published 1 s windows (first insertion t0 = 50 s, second insertion t0 = 90 s) could not be re-downloaded (HTTP 403), so they are not re-run; their published numbers should be treated as deprecated (sync-file offsets: -0.73 ms and -0.87 ms, session minus probe).

## Remaining uncertainty
Sorter spikes are not ground truth; three windows of 1 s on two insertions are a small sample; the control reproduces the published numbers to within one spike (a one-spike edge difference in one window arises from reconstructing the original clock with a constant offset). Chance level is estimated with two circular shifts (41 ms, 73 ms).
