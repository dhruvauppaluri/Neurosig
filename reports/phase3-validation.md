# Phase 3 validation — 2026-11-03

Scope: real-recording Python detection, interpolated/noise-aware localization, IBL spatial estimates, spatial decoding, dashboard method list. RTL, synthesis, and PPA were not modified.

## Commands and outcomes

| Command | Outcome |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tests` | 20 passed |
| `.venv/bin/python -m unittest discover -s edge/tests` | 52 passed |
| `.venv/bin/python -m unittest discover -s viz/tests` | 21 passed (after viz extras) |
| `cd viz/web && npm test && npm run build` | 4 tests passed; TypeScript/Vite build passed |
| `iverilog` `spike_detector_channel_tb` | PASS (82 checks) |
| `iverilog` `spike_event_merge_tb` | PASS |
| `.venv/bin/python -m edge.benchmark_v3` | 324 shared synthetic events × 5 methods; MEArec skipped (no smoke HDF5) |
| `.venv/bin/python -m edge.decode.experiment_v2` | Three shared synthetic splits, A/B/C/D |
| `bash scripts/setup_ibl.sh` | Isolated `data/ibl-venv` |
| `PYTHONPATH=. data/ibl-venv/bin/python -m edge.ibl` | One-second 30 kHz AP chunk plus improved detector |
| `PYTHONPATH=. data/ibl-venv/bin/python -m edge.ibl_eval` | Three disjoint 1 s windows |
| `PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.ibl_experiment` | 358 eligible visual trials, three shared splits |

RTL SystemVerilog sources were not edited. Existing directed detector and merge testbenches still pass.

## Why the original real-data detector failed

On insertion `4836a465-c691-4852-a0b1-dcd2b1ce38a1`, session `caa5dddc-9290-4e27-9f5e-575ba3598614`, probe00, t=100–101 s, 32 VISp contacts:

- Native rate is 30 kHz; the 300–6000 Hz IIR is valid below the 15 kHz Nyquist.
- After median CAR, sorter-aligned troughs have **median 3.42σ**; **7.5% exceed 6σ** and **32% exceed 4σ**.
- The MEArec-calibrated negative 6σ threshold therefore misses most sorter events even before merge.
- Common-mode energy inflates MAD: no-CAR median σ on this chunk is higher than the CAR path (~8 µV after CAR).
- Spike polarity on this selection is predominantly negative; dual polarity was not required for the main gain.
- Sorting is an incomplete reference (missed units, timing jitter, multi-contact assignment).

## Detection before and after (Python only)

Same t=100–101 s chunk, 294 sorter spikes on selected peak channels, 1 ms one-to-one match:

| Detector | Events | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy 6σ, no CAR | 1 | 0 | 1 | 294 | 0.00 | 0.00 |
| CAR + 4σ | 236 | 110 | 126 | 184 | 0.47 | 0.37 |

At 2 ms the CAR/4σ match is 197 TP / 39 FP (precision 0.83, recall 0.67), so much of the 1 ms disagreement is timing, not total miss.

Three disjoint 1 s windows (t=50, 100, 200 s), mean 1 ms recall versus sorting: **0.034 legacy** vs **0.490 CAR/4σ** (range 0.329–0.766). Window t=50 is easier (recall 0.766, precision 0.649); t=100 and t=200 remain harder. Thresholds are estimated per window from the voltages, without labels.

This is coincidence with Kilosort, not ground-truth accuracy, and not an RTL result.

## Synthetic localization (v3)

324 held-out events, seeds 20261103–05, identical voltages per method. Test xyz are offset from the template catalog.

| Method | Median µm | p90 µm | Worst µm | Median runtime ms |
| --- | --- | --- | --- | --- |
| Amplitude | 16.6 | 51.4 | 96.8 | 1.29 |
| Template | 18.7 | 41.5 | 69.4 | 0.18 |
| Interpolated scan (beamform) | 16.4 | 43.2 | 70.3 | 0.41 |
| Weighted scan | 16.8 | 52.2 | 267.2 | 1.04 |
| Robust scan | 18.5 | 59.6 | 273.0 | 1.88 |

v2 medians were 41.3 / 24.2 / 43.2 / 35.0 µm. The large drop is mostly **aligned trough amplitudes** (one time sample, not independent channel minima) plus **top-k interpolation** on the scan. Template matching no longer has a large median lead; it still has the best p90 and lowest runtime.

Overlap remains the failure mode for weighted and robust scans. Electrode dropout (24/32 channels kept, 24 trials) and a 16 µm pitch array (3 sources the engines observe at construction) are in `reports/localization-benchmark-v3.json`. Seed-median ranges are <3 µm for the catalog/scan methods.

**Edge recommendation:** template matching. Interpolated beamform is the catalog-free alternative. Do not treat these micrometers as real-neuron error.

## Real Neuropixels localization

Sorter-triggered snippets, axial disagreement to sorter peak-channel depth (not neuron coordinates). Window t=50 s, 80 spikes:

| Method | Median axial disagreement µm | Within-unit spread µm (14 repeat units) |
| --- | --- | --- |
| Amplitude | 55.2 | 31.5 |
| Template | 117.6 | 58.0 |
| Beamform | 86.0 | 49.9 |
| Weighted scan | 44.9 | 37.0 |
| Robust scan | 71.9 | 39.1 |

Amplitude and weighted scan stay closer to the peak contact, which is expected when the reference *is* a contact. Template/scan estimates spread farther and should not be scored as “wrong.” Method-to-method disagreement is tens of micrometers. Nearby populations on a 20 µm pitch are not reliably separated on this proxy.

## Decoding

Synthetic A/B/C/D mean accuracy: 81.38% / 81.38% / 81.64% / 81.25%. Spatial features do not help this count-driven task.

IBL visual left vs right, 358 trials, 108 held-out per seed, majority 57.41%:

| Model | Mean accuracy |
| --- | --- |
| A counts | 70.37% |
| B counts + electrode geometry | 70.68% |
| C counts + estimated locations | 70.99% |
| D 16-unit MLP on spatial features | 72.22% |

McNemar C vs A and D vs A: p=1.0, few discordant trials. Bootstrap 95% intervals on the accuracy delta include 0. Localization does **not** provide statistically meaningful information beyond spike counts on this session. Occupancy previously collapsed because bins were 0–620 µm while IBL y is ~2700–3000 µm; that bug is fixed, and the conclusion still holds. Likely reasons: 32 VISp units already encode side by identity/rate; estimated locations are transductive cluster-mean waveforms; labels are stimulus side, not reconstructed images.

## RTL

No SystemVerilog changes. The MEArec 6σ golden model is unchanged. CAR, 4σ calibration, and artifact veto would require a new preprocessing contract before any RTL work.

## Remaining limits

- One IBL insertion; three time windows, not independent animals.
- No MEArec smoke file in this environment, so the biophysical nine-spike check did not rerun.
- Robust/weighted scans still fail on overlapping synthetic sources.
- Detector scores versus sorting mix true misses, sorter omissions, and timing jitter.
- No synthesis, timing, area, or power claim.

Reports: `ibl-pilot.json`, `ibl-detection-localization.json`, `ibl-decode.json`, `localization-benchmark-v3.json`, `edge-decode-v2.json`.
