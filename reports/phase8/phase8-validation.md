# Phase 8 validation: localization optimization

Software only. No RTL, FPGA, detector or fixed-point code changed. `edge/localize/*` is unchanged; all new code is in
`edge/phase8/`, `edge/decode/phase8_ablation.py`, `scripts/phase8/`, `edge/tests/test_phase8.py`.
Experiment registry: `reports/phase8/experiment_registry.jsonl` (13 algorithmic experiments, not 30: the search stopped
by rule 3/5 below). Numbers are from the files named in each section; synthetic errors are not real-neuron accuracy.

## Protocol
* **Metric.** Folded error in µm: a collinear probe cannot separate x from z, so truth `(x,y,z)` is scored as `(hypot(x,z), y)`
  for every method (E00). Large error means > 50 µm.
* **Pools.** `dev` (calibration/tuning), `select` (selection; seen), `final` (touched once; `edge/phase8/final_eval.py`
  refuses a second run). Real data: sessions A–D (Phase 5 calibration/selection windows) for development, Phase 6 locked
  sessions E–H run once with the frozen thresholds in `reports/phase8/frozen_real_accept.json`.
* **Baseline reproduced** (`reports/phase8/baseline_v3_reproduction.json`): the 324-event v3 set matches the stored report
  (p90 and rejection counts identical; beamform and robust medians differ by 0.08 µm, all others identical). Newly recorded:
  p95 = 49.5 (template) / 49.4 (beamform) / 58.7 (robust, 42 rejected) / 67.5 (weighted) / 65.9 (amplitude) µm; traced memory 43 MiB.
  That set is half contaminated by a second source and has only 18 distinct truth layouts, so it is a reference, not the target population.

## Goal A: synthetic accuracy (final pool, n=450, 50% contain an interfering source, noise 2/10/25 µV)
| Method | Accepted | Median | p90 | p95 | Large-error rate (all events) | Median runtime |
|---|---:|---:|---:|---:|---:|---:|
| amplitude | 100% | 14.2 | 46.6 | 62.0 | 8.7% | 0.62 ms |
| template | 100% | 18.3 | 41.0 | 49.2 | 4.9% | 0.04 ms |
| beamform | 100% | 18.0 | 44.7 | 53.6 | 6.7% | 0.17 ms |
| robust_scan | 89.3% | 16.7 | 48.4 | 61.6 | 8.4% | 0.76 ms |
| refined_scan (new, K=1) | 100% | 12.3 | 33.4 | 39.4 | 0.9% | 1.27 ms |
| **phase8 (new, K≤2, calibrated)** | **93.8%** | **5.0** | **29.6** | **40.4** | **2.0%** | 10.3 ms |

Phase 8 by subset (accepted events): single source 2.3 / 11.2 / 13.9 µm (median/p90/p95); contaminated 15.2 / 40.4 / 48.9; 25 µV noise 15.7 / 38.9 / 42.9 (82% accepted).
Paired median difference on the accepted events vs beamform +12.0 µm (95% CI 10.6–13.7), vs template +12.3 (10.7–14.0).
Largest levers (dev, E01–E03): whitened temporal-rank-1 amplitudes instead of a single trough sample (25 µV median 36.9 → 18.8), then continuous LM refinement (single-source median 4.0 → 1.8).
* Median ≤ 12 µm: **met** (5.0; refined_scan alone 12.3 narrowly misses). p90 ≤ 30 µm: **met narrowly** (29.6; on `select` it was 32.3, so treat as borderline).
* p95 ≥ 20% below best baseline p95 (template 49.2): phase8 −17.9%, refined_scan −19.9%: **narrowly missed**.
* Runtime ≤ 2× comparable method: **refined_scan 1.57× robust_scan (met); full phase8 14.4× (not met)**. See Goal F.
* **Model mismatch** (events rendered with a line-source gain, scans keep the monopole gain; n=180): phase8 accepts 90%, median 19.0, p90 54.4 µm, region coverage 77.8%; amplitude gets 16.8/41.8, refined_scan 17.3/34.7. The gains above depend on the forward model matching the data.

## Goal B: overlapping sources (`overlap_bench`, final pool: 120 singles, 480 pairs, 120 triples)
Resolvable pair (fixed in code first): y-separation ≥ 60 µm, both peak-channel SNR ≥ 6, amplitude ratio ≥ 1/3. One-to-one Hungarian scoring, 60 µm gate.
| | beamform | template | robust_scan | phase8 |
|---|---:|---:|---:|---:|
| Two-source recall, resolvable pairs | 0 | 0 | 0 | **89.2%** (target ≥ 90%: missed by 0.8 pt) |
| Recall, all pairs | 0 | 0 | 0 | 61.6% |
| False two-source rate, singles | 0 | 0 | 0 | **5.0%** (synthetic final pool: 5.7%; target ≤ 5%: borderline/missed) |
| Per-source median error, correctly resolved pairs | – | – | – | **3.3 µm** (p90 13.7) |
| Difficult-event p90 (|Δt| ≤ 4 samples, sep ≤ 120 µm) | 63.8 | 62.1 | 66.9 | 59.1 (**−4.9% vs template; target −25% missed**) |
| Rejected (all / resolvable) | 0 / 0 | 0 / 0 | 13.5% / 9.2% | 9.2% / 2.3% |
| Triples: sources found (of 3) | 1.0 | 1.0 | 0.78 | 1.7 |
On `dev` the same pipeline reached 95.3% recall, 1.7% false-two and −27% difficult p90; those did not hold on the final pool (small difficult subset, thresholds calibrated on a different generator). Missing a second source is a *silent* failure at high noise: at 25 µV almost no pair is detectable, and those events are flagged only through the calibrated region size.
Two-source detection uses (i) a free-waveform SSE-drop test and (ii) a shared-waveform rank-1 test (null ≈ χ²₃) for near-simultaneous pairs (E05), with same-sign and boundary guards (E06).

## Goal C: real Neuropixels (proxies; no neuron ground truth)
Held-out sessions E–H (24 locked windows, 1,439 sorter events, 113 units with ≥ 5 events; `reports/phase8/real_stability_final.json`):
| | beamform | template | robust_scan | phase8 (K=1) |
|---|---:|---:|---:|---:|
| Usable coverage | 100% | 100% | 80.7% | 92.1% |
| Within-unit xy spread, median µm | **19.3** | 25.7 | 28.2 | 26.0 |
| Axial chance ratio (within/total MAD) | 0.056 | 0.076 | 0.079 | 0.099 |
| Early-vs-late drift, median µm | **15.9** | 17.2 | 19.8 | 105.5 |
| Dropout 25% / 50%: displacement, µm | 5.8 / 21.0 | 9.4 / 21.1 | 12.0 / 26.3 | 12.7 / 33.2 |
| Dropout failure rate (25% / 50%) | 0 / 0 | 0 / 0 | 8.0% / 5.1% | 0 / 0 |
| Noise ×1.0 displacement, µm | **3.8** | 5.3 | 8.8 | 8.1 |
Phase 8 does **not** improve real-data stability: within-unit spread is 35% *worse* than beamform (target: ≥ 25% better), displacement under dropout is worse, and failures were already zero for beamform/template (the ≥ 20% failure-reduction target is not applicable against them; against robust_scan it is a reduction from 8% to 0%). Usable coverage 92.1% (≥ 90%) holds on held-out sessions with thresholds frozen from the development windows. Session E–H per-session spread for phase8 ranges 14.9–54.7 µm.
Why (dev, E10–E12): the free-waveform fit uses positive lobes and neighbouring units' spikes that a same-sign monopole cannot represent; fits run to the array end. Clipping, robust loss, no-SVD, model-error inflation and spatial windows did not help (best 27.6 vs beamform 9.9 µm dev spread); a weak prior toward the strongest contact approached beamform only by anchoring to the peak contact, which the stability proxy rewards, so tuning stopped (rule 3). The quantized, peak-anchored scans look stable partly because they are anchored; a stability win would not have been evidence of better localization.
Repeated-unit spread, dropout, noise, temporal drift and cross-session (E–H vs development) are implemented. Channel-spacing variation is covered only on synthetic arrays (missing contacts test). Recording-duration variation and brain-region diversity beyond the eight insertions were not tested.

## Goal D: calibrated confidence (frozen calibration `b52092afcb4f`)
Returned per event (`LocalizationResult.to_dict`): coordinates for each source, 2×2 covariance, 90% region area, confidence (1 − size/cap), quality flags, accepted flag, reject reason, χ²_red, SSE-drop statistics.
Mondrian split-conformal quantiles by stratum (clean / flagged / two-source / shared-waveform), accepted-only calibration, area cap at 92% dev acceptance, empirical χ² cap.
* Coverage of nominal 90% regions (final pool, accepted events): **91.2%** overall (target 90 ± 3: met). Conditional: singles 100% (conservative), **contaminated 82.5%**, 25 µV 90.2%. Under model mismatch **77.8%**.
* Acceptance 93.8% (≥ 90%: met). Median 90% region area 2,129 µm² overall (sharpness: ≈ ±26 µm equivalent radius; single-source events far smaller, 25 µV events much larger, see json).
* Large-error (> 50 µm) rate 2.0% of all events vs 4.9–8.7% for baselines (−59% vs template) at 93.8% acceptance. Risk–coverage (ranking by uncertainty): p90 18.6 / 23.5 / 29.5 / 30.7 µm at 50/70/90/100% kept, versus beamform confidence ranking 30.5 / 32.5 / 38.3 / 44.7 (`final_eval.json`).
* Calibration is therefore marginally valid on the matched model but not conditionally valid for undetected overlaps, and not valid under forward-model mismatch.

## Goal E: decoding ablation (8 insertions, `reports/phase8/decode_ablation*.json`)
32 units from the most populated brain region per insertion (region chosen by unit count, not by visual responsiveness), counts 50–250 ms after stimulus, repeated 5-fold × 4 stratified CV, fixed hyperparameters, identical folds. Locations are placed from cluster mean waveforms without labels (still session-global).
| Mean over 8 sessions | Accuracy |
|---|---:|
| Majority | 52.92% |
| Counts | 59.11% |
| + electrode coordinates | 59.10% |
| + original robust_scan locations | 58.96% |
| + Phase 8 locations | 58.89% |
| + Phase 8 locations + uncertainty | 58.87% |
| Shuffled-position control | 59.07% |
| MLP counts / MLP Phase 8 + uncertainty | 59.42% / 59.14% |
Paired deltas (percentage points, bootstrap over sessions): Phase 8 locations vs counts −0.22 [−0.43, +0.02]; vs original locations −0.07 [−0.49, +0.32]; with uncertainty vs counts −0.24 [−0.40, −0.07]; improvement ≥ 3 pt target **not reached**; positive in 1 of 8 sessions; only session H is individually significantly negative (−0.76, CI −1.4…−0.2).
Interpretation: improved localization added no decodable information here. With fixed per-unit positions, occupancy and mean-position features are linear in the counts, so a linear decoder spans them already; a gain would need nonlinear use or regularization, and the MLP control did not show one. Six of eight sessions decode near chance because the most populated region is not visual; a visually-selected re-run would be a new protocol.

## Goal F: efficiency (`reports/phase8/runtime_bench.json`)
Single-process Python on this laptop: median/p95/p99 per event: beamform 0.15/0.32/0.50 ms, robust_scan 0.72/0.85/1.04, refined_scan 1.13/1.40/1.88, **phase8 10.5/14.2/17.3 ms (14.4× robust_scan, ≈ 96 events/s/core)**. Cost is scipy call overhead (two LM refinements plus a Powell fit and a numeric covariance); rank-4 temporal reduction and a grid prescreen gave ≤ 20% (E07–E08, prescreen rejected). Channel scaling (median ms at 16/32/64): refined_scan 1.2/1.3/2.0, phase8 12.4/13.0/14.0. Peak traced memory ≈ 0.9 MiB (phase8), 1.2 MiB (refined), 5 MiB (robust_scan).
Burst simulation (deterministic, queue 256, single worker, measured service times): at 300 events/s phase8-only drops 24.5% (p99 delay 2.5 s); tiered policy (fall back to refined_scan when the queue holds ≥ 16) drops 0% (p99 131 ms); at 1000 events/s tiered drops 4.2%. The assumed event rate (10–300/s for 32 channels) is not measured. No batch speed-up was measured.
Pareto: **refined_scan** (1.6× robust_scan, no rejection, p95 39.4 µm) is the practical edge candidate; the full pipeline buys overlap resolution and calibrated regions at ~14× cost.

## Goals reached / missed
| Goal | Status |
|---|---|
| A median ≤ 12 µm | Met (5.0) |
| A p90 ≤ 30 µm | Met narrowly (29.6; 32.3 on select) |
| A p95 −20% | Narrowly missed (−17.9%; refined_scan −19.9%) |
| A runtime ≤ 2× | Met only for refined_scan; full pipeline 14.4× |
| B recall ≥ 90% resolvable | Missed (89.2%; dev 95.3%) |
| B false two-source ≤ 5% | Borderline (5.0% overlap bench, 5.7% synthetic final) |
| B per-source median ≤ 20 µm | Met (3.3) |
| B difficult p90 −25% | Missed (−4.9% final; −27% dev) |
| C within-unit −25%, dropout −20% | Missed (worse than beamform; failures were already 0 for beamform) |
| C usable coverage ≥ 90% | Met (92.1%) |
| D 90% region coverage | Met overall (91.2%); not conditionally for overlaps (82.5%) or mismatch (77.8%) |
| D large-error −25% | Met (−59% vs template at 93.8% acceptance) |
| E +3 pt decoding | Missed (−0.2 pt); null result, reason given |
| F ≤ 2× runtime, burst stability | refined_scan met; full pipeline met only with tiered fallback |

## Stopping condition and next phase
Stopped under rules 3 and 5: real-data repairs stopped improving the development objective without anchoring, and the decoding question has a structural answer. The final holdout was evaluated once; do not tune against it.
Recommended next phase (new protocol): (1) a real-data-faithful forward model (line/extended source, measured spatial decay, probe-specific geometry) and calibration on it, since synthetic gains do not transfer; (2) real two-unit ground truth (paired recordings or MEArec/LFPy with true morphology) for the overlap detector; (3) conditional conformal calibration by SNR so that undetected overlaps do not under-cover; (4) a native-code or analytic-Jacobian fit to bring the full pipeline under ~2 ms; (5) a decode protocol that selects visually responsive units a priori and lets position inform regularization/pooling rather than adding a linear feature.

## Reproduce
```sh
.venv/bin/python -m edge.phase8.baseline
PYTHONPATH=. .venv/bin/python scripts/phase8/exp_a.py            # Goal A dev
PYTHONPATH=. .venv/bin/python -m edge.phase8.final_eval         # select + final (once)
PYTHONPATH=. .venv/bin/python scripts/phase8/real_stability.py --splits dev
PYTHONPATH=. .venv/bin/python scripts/phase8/real_stability.py --splits final --config reports/phase8/frozen_real_accept.json
PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase8_ablation --pids "A=4836a465-c691-4852-a0b1-dcd2b1ce38a1,..."
PYTHONPATH=. .venv/bin/python -m edge.phase8.runtime_bench
.venv/bin/python -m unittest edge.tests.test_phase8
```
Versions: Python 3.12.15, numpy 2.5.4, scipy 1.18.1; code revision recorded per experiment in the registry.
