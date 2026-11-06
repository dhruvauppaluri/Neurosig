# Phase 5 validation: real-signal spike detection and fixed-point reference

Branch `phase5-realsignal-fixedpoint`. No SystemVerilog, FPGA or historical report file was modified.
All "agreement" numbers compare against the IBL pykilosort spike sorter, which is **not biological ground truth**.

## Summary

1. The improved detector (`detector-v2.0`, band 200–4500 Hz instead of 300–6000 Hz, everything else unchanged) is a **small but consistent** improvement:
   on 20 unseen windows (4 sessions, one never seen before) channel-consistent F1 at 1 ms rose 0.537 → 0.553
   (paired 95 % interval +0.012 to +0.022; better in 19/20 windows; positive in every session). Recall is unchanged (48.8 % → 49.0 %); the gain is fewer unmatched detections (false detections/s −15 %).
2. Two methodological problems in earlier phases were found and fixed for this phase: sorter times were compared with raw samples without the probe-to-session clock sync (up to 31 ms apart), and unconstrained 1 ms matching has a high chance level (~37 %).
3. A bit-specified integer reference (`fixed-v1`) exists, with two independent implementations that agree exactly, and reproduces 98.4 % of the float events exactly on unseen data. No RTL implements it.

## Data and protocol

Four public IBL insertions: A `4836a465` (steinmetzlab), B `1a60a6e1` (NYU-45), C `7a620688` (UCLA, VIS*), D `b114e7e0` (hausserlab, VISp). 32 contiguous visual-cortex contacts, native 30 kHz, 2 s windows, seeded draws
(`reports/phase5/split_manifest.json`). Calibration (12 windows) was used for diagnostics only; selection (12 windows from A, B, C) chose the detector under rules written before the run (`selection_criteria.json`);
held-out (20 windows: 4 each from A, B, C and 8 from D, which was never used for anything else) was run once, after the detector file was frozen and committed.
Two earlier calibration sessions' 1 s windows (50–200 s) were used only to reproduce old numbers. Session A's windows were redrawn once (recording is 2171 s, not 5000 s) before any result was viewed.
Four insertions is still small; labs and probes differ, but not enough to claim broad generalisation.

## Finding 0: corrections to the earlier measurement

* **Clock sync.** IBL `spikes.times` are on the session clock; raw samples are on the probe clock, and the offset grows roughly linearly (−0.5 ms at 50 s, −3 ms at 326 s, −31 ms at 3259 s in session B). The Phase 3–4 evaluation compared them directly.
  With the sync applied, the Phase 4 baseline's recall on the first-insertion 100 s and 200 s windows is 0.629 and 0.637 (previously reported as 0.374 and 0.329; 1 ms, unconstrained). The old 49.0 % / 46.9 % figures were therefore understated for later windows. Old numbers were reproduced exactly before the correction (206/63/136, 110/126/184, 96/144/196 TP/FP/FN).
* **Chance level.** With ~270 events/s on 32 channels, shifting the reference by 41 ms and 73 ms still matches ~37–39 % of reference spikes at ±1 ms (unconstrained). The primary metric here is therefore *channel-consistent* (detection within 60 µm of the sorter's peak contact), and chance is reported next to everything.
  Held-out baseline: unconstrained recall 58.2 % (chance 38.7 %), channel-consistent F1 0.537 (chance 0.194).

## Diagnosis: why the CAR/4σ detector misses reference spikes (calibration windows, 6257 reference spikes)

| Outcome | Share | Interpretation |
|---|---|---|
| Matched (1 ms, unconstrained) | 62.6 % | |
| Sub-threshold: trough 3–4σ | 10.7 % | **Detection failure** — the sorter found a spike whose trough is below the 4σ threshold |
| Sub-threshold: trough < 3σ | 1.6 % | Detection failure, probably not detectable by thresholding |
| "Timing-like": detection on a nearby contact within 1–5 ms | 15.2 % | Mixed: of a sampled 746, 24 % have ≥ 4σ trough at the reference time (genuine **timing / peak-choice** differences, ≈ 3.7 % of all references), 71 % have a 3–4σ trough next to another detection (sub-threshold spike in a burst), 5 % < 3σ |
| Removed by per-channel 1 ms refractory | 7.7 % | Reference spike within 1 ms of another event on the same contact (bursts, adjacent units) |
| Removed by merge or veto | 0.8 % | Merge/veto is not a significant loss at 32 channels |
| Crossing present but peak search missed it | 1.3 % | |

So roughly 21 % of reference spikes are sub-threshold 3–4σ events (directly and inside the "timing-like" bucket), ~8 % are lost to the refractory rule, and ~4 % are real timing/peak disagreements.
Timing is otherwise tight: the matched channel-consistent error has median 0.0 ms and MAD 0.033 ms (1 sample); there is no causal-filter latency bias. A small excess of detections at +1.0 to +1.6 ms after a reference is unexplained (candidates: a second unit, an after-potential).
**Reference disagreement** (detections without a sorter match, calibration): 1324 are within 5 ms and 60 µm of a reference spike, 747 within 5 ms but elsewhere on the probe, 427 isolated and weak (< 6σ), 27 isolated and strong, 14 multi-channel. These are not called false positives; the sorter does not list low-amplitude or multi-unit activity.
Per-channel noise varied up to a few-fold between contacts, which the per-channel σ already handles; common-mode noise is the largest single effect (dropping CAR cost ΔF1 −0.08). Figures: `reports/phase5/figures/`.

## Detector experiments (selection split, 12 windows, channel-consistent F1 at 1 ms; baseline 0.455, chance 0.167)

Rule: 95 % window-cluster bootstrap interval of ΔF1 > 0 and ΔF1 ≥ 0.01, no session worse than −0.02, false detections/s ≤ +25 %, runtime ≤ 3×. Full table: `benchmark_selection_singles.json`.

| Change | ΔF1 [95 % CI] | Verdict |
|---|---|---|
| High-pass 200 Hz | +0.014 [+0.001, +0.031] | **adopted** |
| then low-pass 4500 Hz (vs hp_200) | +0.011 [+0.003, +0.018] | **adopted** |
| High-pass 150 / 100 Hz (vs hp_200) | +0.004 / +0.007, CI incl. 0 | not adopted |
| Amplitude cap 30σ | +0.014 [0.000, +0.042] | not adopted (borderline) |
| Causal EMA noise estimate | +0.014 [−0.012, +0.054], one session −0.020 | inconclusive |
| MAD, IQR, clipped-RMS noise | −0.00 … +0.012, all CIs incl. 0 | no effect |
| Width gate ≥ 2/3 samples, artifact fraction 0.25/0.75/off, peak search 0.25/1.0 ms, refractory 0.5/2 ms | ≤ ±0.004 | no effect |
| Spatial combination of adjacent contacts (w 0.5 / 40 µm) | +0.006, CI incl. 0 | no benefit; wider/stronger: −0.017 … −0.044 |
| Merge: NMS 60 µm / 100 µm | −0.008 / +0.008, false detections/s +71 % / +32 % | worse trade-off |
| Merge window 1.0 ms | −0.018 | worse |
| **No CAR** | **−0.082 [−0.172, −0.016]** | clearly worse |
| Mean CAR | −0.006 [−0.022, +0.009], one session −0.035, false detections/s −23 % | not adopted |
| Threshold 3.5σ / 4.5σ / 5σ | −0.047 / −0.014 / −0.048 | worse (3.5σ doubles false detections/s) |
| High-pass 400 / 500 Hz, low-pass 9000 Hz | −0.009 / −0.018 / −0.014 | worse |

Consistent helpers: median CAR (vs none) and a narrower band. Nothing else helped on its own. Change in selection F1, baseline → v2: 0.455 → 0.480 (+0.025); the held-out gain is smaller (+0.017), as expected for a choice made on the selection set.

## Held-out result (20 windows, 15 366 reference spikes, evaluated once)

| Detector | tol (ms) | chan.-consistent F1 [95 % CI] | recall | precision | chance F1 | false det./s |
|---|---|---|---|---|---|---|
| baseline | 0.5 / 1 / 2 | 0.508 / 0.537 / 0.589 | 0.462 / 0.488 / 0.535 | 0.565 / 0.597 / 0.656 | 0.113 / 0.194 / 0.310 | 136 / 126 / 108 |
| **v2** | 0.5 / 1 / 2 | **0.529 / 0.553 / 0.599** [0.527, 0.577 at 1 ms] | 0.469 / 0.490 / 0.531 | 0.608 / 0.636 / 0.688 | 0.109 / 0.192 / 0.305 | 116 / 108 / 92 |
| v2 + mean CAR | 0.5 / 1 / 2 | 0.519 / 0.539 / 0.575 | 0.421 / 0.438 / 0.467 | 0.675 / 0.700 / 0.747 | 0.098 / 0.173 / 0.278 | 78 / 72 / 61 |

v2 − baseline at 1 ms: ΔF1 +0.0166 [+0.0117, +0.0224], false detections/s ×0.85, runtime ×0.95; per session +0.043 (A), +0.014 (B), +0.011 (C), +0.014 (D, never seen in selection). Timing error (matched, 1 ms): median 0.0 ms, 5–95 % −0.17 / +0.30 ms for v2.
Regressions: none for v2 on held-out. For mean CAR: recall −5.0 points, session B −0.023, D −0.009.
Unconstrained numbers (comparable to earlier phases) are in `benchmark_heldout.json` (v2 recall 57.7 %, chance 37.0 %; baseline 58.2 %, chance 38.7 %): on that metric v2 is **not** better in recall.

## Float vs fixed (held-out, v2; `float_vs_fixed_heldout_v2.json`)

Both paths ran on the same ADC codes (the data are already exactly 2.34375 µV quantised, so ADC quantisation error is 0).

| Stage | Measured difference (max over windows) | Cause |
|---|---|---|
| Filter | RMS 0.24 µV (≤ 0.035σ), max 0.87 µV | Q2.30 coefficients, 24-bit states, truncation toward zero |
| CAR | max 0.85 µV | follows the filter error; integer even-N median floors the midpoint |
| Baseline / threshold | threshold max 0.12 µV; σ relative error ≤ 0.37 % | integer MAD, Q16 constants, rounding |
| Candidates (26 267 float) | 25 922 identical (98.7 %); 345 float-only, 299 fixed-only; 157 of 26 079 matched peaks differ by 1–3 samples | threshold-boundary samples and near-tied minima |
| Events | 11 657 of 11 846 float events reproduced exactly (98.4 %); 165 integer-only events; vetoed groups 39 = 39 | cascades of the above into merge winners; timestamp offsets up to 15 samples occur only when the merge winner differs |
| Normalised amplitude | RMS ≤ 0.023σ on identical events | Q8.8 and reciprocal rounding |
| Sorter agreement | 7513 vs 7530 matched (−0.2 %) | |
| Saturation / overflow | none; accumulator ≤ 50 bits | |

The integer model is *specified*, not claimed equal to float. Deterministic: two implementations (vectorised numpy and scalar frame-by-frame) are bit-identical on random streams with disabled channels and dropped frames; outputs have no data-dependent ordering. 10 guard bits are essential (see spec table; 4 bits gave 3–6 µV filter error and only 74 % identical candidates).
Mean-CAR variant in fixed point: 9520 of 9599 float events reproduced exactly.

## Tests and complexity

* New tests: 39 (`tests/test_phase5_detectors.py`, `tests/test_phase5_fixed.py`): baseline equivalence, optimal matching, every corner case on the brief's list, 5 seeded randomized stream-equivalence runs, golden vectors, stability and zero-input decay for both coefficient sets. Full suite (`pytest tests edge/tests`): 115 passed (76 pre-existing + 39 new).
* Cost (analytical, per 32-channel frame): filter 2 biquads × 5 multiplies per channel (unchanged by v2); median CAR 191 compare-exchanges (Batcher) or a serial rank select; mean CAR 31 adds; detection 1 comparison per channel. Python runtimes: 6 s per 20 windows for baseline and v2 alike (not a hardware metric). v2 adds **no** hardware beyond new coefficients.

## Answers

* **Why does CAR/4σ still miss spikes?** About a fifth of the sorter's spikes have 3–4σ troughs below the 4σ threshold; ~8 % fall inside a 1 ms refractory interval of another event on the same contact; ~4 % are real peak/timing disagreements; the rest of the apparent gap was the clock offset and chance matching in the old metric. Lowering the threshold to 3.5σ raises recall but doubles false detections and lowers F1.
* **What helped consistently?** Median CAR (vs none), and a narrower 200–4500 Hz band (+0.017 held-out). **What hurt?** No CAR (−0.08), thresholds ≠ 4σ, higher high-pass corners, 9 kHz low-pass, longer merge window, spatial combination at 60 µm, NMS merging (false detections +32–71 %).
* **Improvement on unseen recordings:** +0.017 F1 (0.537 → 0.553) at 1 ms, CI +0.012…+0.022, 19/20 windows, all four sessions; recall flat; precision +3.8 points. Modest, and relative to the sorter.
* **Fixed vs float:** 98.4 % of events identical; sorter agreement within 0.2 %; differences come from filter truncation and threshold-boundary samples. **Deterministic and fully specified?** Yes for ADC frames → events, given calibration registers; timestamp wrap and event-queue depth are not specified.
* **RTL changes needed:** frame assembler + CAR stage, 30 kHz 200/4500 coefficients and the 10-bit input shift, σ-normalised merge key, artifact veto, valid/active masks, calibration register file (`docs/phase5-rtl-readiness.md`).
* **Limitations:** 32 contacts per insertion and four insertions; sorter reference excludes low-amplitude activity so precision is an agreement rate; calibration used the evaluated window itself (cross-window calibration gave the same F1, one pass, informational); mean-CAR vs median trade-off measured at one threshold; the recordings are 10-bit, so behaviour with other ADC resolutions is untested; no synthesis, timing or area; the Phase 4 results in `reports/ibl-*.json` were not rewritten and still carry the uncorrected sync.
