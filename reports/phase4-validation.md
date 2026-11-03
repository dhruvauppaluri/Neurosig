# Phase 4 validation — 2026-11-03 (America/Los_Angeles)

Scope: freeze the Python real-recording detector, fail-closed overlap localization, a second public IBL insertion, decode from detector events, and restore the MEArec/RTL 9/9 path. No SystemVerilog functional changes. No synthesis or PPA.

## Commands and outcomes

| Command | Outcome |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tests` | 21 passed |
| `.venv/bin/python -m unittest discover -s edge/tests` | 55 passed (overlap fail-closed + VIS fallback) |
| `.venv/bin/python scripts/freeze_real_detector.py` | Wrote `reports/frozen-real-detector.json`; `rtl_follows` is false |
| `.venv/bin/python -m edge.benchmark_v3` | 324 synthetic events; MEArec 9 spikes present |
| `PYTHONPATH=. data/ibl-venv/bin/python -m edge.ibl_eval --skip-primary` | Second insertion two 1 s windows |
| `PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.detector_experiment` | 40 s AP blocks, t0=1465 s / 1825 s |
| Verilator 5.020 full pipeline | PASS: 1,024,000 ADC samples, 9 exact merged events |
| `iverilog` `spike_detector_channel_tb` | PASS (82 checks) |
| `iverilog` `spike_event_merge_tb` | PASS |
| MEArec float detector / fixed-point | 9 TP, 0 FP, 0 FN on the restored seed-42 recording |

## Frozen preprocessor

Contract name: `car_4sigma_300_6000_negative`. Median CAR, 300–6000 Hz causal IIR, 4σ negative threshold, iterative MAD, 0.5 artifact-channel fraction. Selected from the IBL variant table: most stable 1 ms F1 versus sorting without dual-polarity false-positive inflation. A 9 kHz low-pass was similar; 6 kHz is frozen to match the hardware IIR band.

**RTL does not follow.** Median CAR is required before the 4σ comparison and is absent from `rtl/neurosig_pipeline.sv`. Loading 4σ threshold registers without CAR would not reproduce this Python detector.

## Overlap localization

`edge/localize/overlap.py` flags a bimodal aligned-trough profile (`scipy.signal.find_peaks`, height 0.85×peak, ~80 µm separation). `RobustScanLocalizer` returns `available=False` instead of a centroid.

On the 324-event v3 synthetic set, robust scan had **42 failures**. Remaining-event median/p90/worst were **16.3 / 47.6 / 101.7 µm** (previous worst with forced estimates was ~273 µm). Weighted scan still reports overlap tails. Template matching remains the recommended method. The MEArec nine-spike check ran: template median 29.5 µm.

## Second IBL session (detection)

Insertion `1a60a6e1-da99-4d4e-a734-39b1d4544fad`, session `51e53aff-1d5d-4182-a684-aba783d50ae5`, NYU-45 2021-07-19 probe00, VIS* contacts (VISa, not VISp). Two 1 s windows (t=50 s, 90 s), 1 ms versus sorting:

| Session | Legacy 6σ mean recall | Frozen CAR/4σ mean recall | Range |
| --- | --- | --- | --- |
| 4836a465 (3 windows) | 0.034 | 0.490 | 0.329–0.766 |
| 1a60a6e1 (2 windows) | 0.081 | 0.469 | 0.336–0.602 |

This is sorter coincidence, not ground truth.

## Detector-event decode

Eligible left-XOR-right visual trials start at 108.8 s on the first insertion, so a t=90 s 16 s block contained **zero** trials (3556 detector events). The reported run uses 40 s AP around a trial-dense region:

- Primary: t0=1465 s, 11 trials, 8065 detector events, majority 63.6%. Held-out (5 trials): sorted-count accuracy 0.60, detector-count accuracy 0.20. McNemar p=0.625; bootstrap 95% CI on detector−sorted includes 0.
- Secondary: t0=1825 s, 10 trials, majority 70%. Train primary / test secondary: detector 0.70 (equals majority), sorted 0.30 (collapsed to one class).

Do not rank detectors from this n. The experiment shows the event-binning path is wired, not that contact events beat Kilosort.

## MEArec / RTL

Restored `data/smoke/recording.h5` (seed 42, 32,000×32, nine ground-truth spikes). Float detector 9/9. Fixed-point 9/9. Verilator 5.020 matched all nine Python merged events exactly (sample, channel, code) and 9/9 versus MEArec with 1 ms matching. Simulator version differs from the November 2 Verilator 5.050 record; the event list is the same.

Reports: `frozen-real-detector.json`, `ibl-detection-localization.json`, `ibl-detector-decode.json`, `localization-benchmark-v3.json`, `rtl-pipeline-validation.json`.
