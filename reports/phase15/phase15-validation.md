# Phase 15 validation: localization, overlap, identity, cross-animal decoding and runtime (software only)

Frozen configuration `reports/phase15/frozen_p15.json`, hash `1bcba1777bd6`, written on 2026-11-09 before any final
data were opened. Single-use final run: `scripts/phase15/run_final15.py` -> `reports/phase15/final15.json` (all 10
steps returned 0). Methods and splits: `docs/phase15-research.md`. Every experiment: `experiment_registry.jsonl` (109
lines). Dashboard: `dashboard_phase15.json` (61 Phase 14 baseline rows + 85 Phase 15 rows, one row per metric).
No RTL, FPGA, frozen detector, fixed-point or `edge/localize/*` code changed. Regression:
`scripts/run_regression.py --no-realdata` 327 passed, 0 failed, 1 skipped (includes 15 Phase 15 tests).

## Read this first

* **Biophysical results (Tracks A, B, uncertainty) are cross-validated, not final.** All 13 BBP layer-5 morphologies
  were already used by Phase 14, so no independent morphology remains. Phase 15 reports leave-one-morphology-out
  (LOMO) results with nested threshold selection and calls them exploratory.
* **Final data:** IBL X1-X4 (4 new insertions, 4 labs) and 6 never-opened Allen mice. X4 contributes nothing to the
  identity, real-count or IBL decoding finals: it has no high-quality sorter unit to score against, no usable
  sorter-aligned segment cache (the session builder returned none) and one visual unit. Those finals cover X1-X3.
  Allen final: 6 mice, trained on 24 (17 Phase 13-14 + 7 new training mice).
* Identity and real-count metrics use sorter labels for scoring only; they are proxies, not ground truth.
* Cross-animal decoding uses Allen sorted units, not raw voltage. Identity classification is not image reconstruction.

## Per-metric comparison

Baseline = frozen Phase 14 result as published. "Same protocol" baselines are the Phase 14 method re-run on the Phase
15 evaluation set, which is the fair comparison wherever the evaluation set changed.

### D. Cross-animal visual decoding (balanced accuracy, 6 final mice, leave-one-mouse-out style: train 24, test each)

| Task (chance) | P14 final (other mice) | P14 spec, same final mice | Best dev (LOMO, 11 mice) | **D-P15 final** | Paired diff vs P14 spec [95% CI], mice improved, sign p | Target | Reached |
|---|---|---|---|---|---|---|---|
| Spatial frequency (0.20) | 0.422 | 0.438 | 0.515 (e7n) | **0.536** (min 0.393) | +0.098 [+0.064, +0.134], 6/6, p 0.031 | 0.60 | no |
| Static orientation (0.167) | 0.232 | 0.218 | 0.206 (e11) | 0.221 | +0.003 [-0.011, +0.013], 4/6, p 0.69 | 0.40 | no |
| Drifting orientation (0.125) | 0.195 | 0.194 | 0.203 (e11) | 0.206 | +0.012 [+0.002, +0.021], 4/6, p 0.38 | 0.40 | no |
| Gabor x (0.111) | 0.191 | 0.179 | 0.173 (e1) | 0.185 | +0.006 [-0.003, +0.015], 4/6, p 0.69 | 0.35 | no |
| Gabor y (0.111) | 0.176 | 0.252 | 0.214 (e10n) | 0.249 | -0.003 [-0.013, +0.008], 2/6, p 0.69 | 0.35 | no |
| Natural image, 118 (0.008) | 0.044 | 0.049 | 0.047 (e11) | 0.052 | +0.003 [-0.001, +0.007], 5/6, p 0.22 | 0.15 | no |

D-P15 = groups CCF area x functional cluster (k-means, k = 6, of each unit's stimulus-averaged response time course,
centres from training mice only), 25 ms bins over 0-250 ms, per-session label-free z-scoring, logistic regression
(C = 0.1); 360 features. The pre-registered secondary per-task routing (e11 MLP on orientation, drifting orientation
and natural images) gave 0.213 / 0.202 / 0.061 on those tasks: no reliable gain over D-P15. Relative improvement on
spatial frequency: +22% over the same-mice reference, +27% over the published P14 number. Coverage: every trial of
every final mouse. Cost: feature construction plus a linear model, under 1 ms per trial.

Learning curve on dev (spatial frequency, area x depth): 2 / 4 / 7 / 10 training mice -> 0.350 / 0.382 / 0.425 /
0.436. The final used 24 training mice; the reference spec rose from 0.422 to 0.438 with them on different mice.

IBL X1-X3 with the frozen Phase 13 protocol: every task is at chance or majority (contrast-5 0.25 vs majority 0.24,
stimulus side 0.58 vs 0.55, presence 0.50 balanced). The Phase 13 timing advantage did **not** replicate:
temporal-vs-plain counts on contrast +0.2 points [-1.8, +3.3], spatial maps vs shuffled positions -1.1 points.

### C. Online identity (IBL X1-X3, sorter-scored; median over sessions, minimum session in brackets)

| Metric | P14 final (T-W) | OU-P14 on X, 2 s | **ID-P15 on X, 2 s** | OU-P14 on X, 30 s | **ID-P15 on X, 30 s** | Target | Reached |
|---|---|---|---|---|---|---|---|
| HQ precision | 0.967 | 1.000 [0.904] | **1.000 [0.909]** | 1.000 [0.986] | **0.974 [0.965]** | >= 0.98 | 2 s yes (median), 30 s no |
| HQ coverage | 0.595 | 0.507 [0.300] | **0.739 [0.520]** | 0.688 [0.656] | **0.849 [0.817]** | >= 0.85 | no (30 s misses by 0.1 pt) |
| All-event coverage | 0.057 | 0.097 | 0.103 | 0.095 | 0.114 | >= 0.40 | no |
| All-matched precision | 0.744 | 0.972 | 0.966 | 0.991 | 0.926 | - | - |
| Merge rate | 0.45 | 0.00 | 0.00 | 0.00 | 0.083 | - | - |
| Unit/event spread reduction | 0.607 | 0.751 | 0.740 | 0.636 | 0.716 | >= 0.80 | no |
| Compute | 0.11 ms/event | 0.026 ms | 0.024 ms | 0.032 ms | 0.028 ms | - | - |

ID-P15 raises HQ coverage by +23 points (2 s) and +16 points (30 s) over the same-session OU-P14, at a cost of
-2.6 points HQ precision and more merges on 30 s windows. Dev / select: HQ coverage 0.646 -> 0.806 / 0.455 -> 0.596,
HQ precision 0.997 -> 0.985 / 0.949 -> 0.911. The trade-off is real: every coverage gain on development came with a
precision loss; the spread-reduction difference between OU-P14 and ID-P15 on the same sessions is within noise.
All-event coverage stays near 10% because 68% of detector events fall below the SNR-6 assignment floor (dev
abstention breakdown); low-SNR likelihood assignment was tried and lowered precision.

### B. Source count / overlap (MEArec, 13 morphologies, LOMO, nested thresholds; NP1 geometry, IBL noise)

| Metric | P14 final (unseen 7 morph.) | P14 F21, same LOMO protocol | **B-P15 (LOMO)** | Target | Reached |
|---|---|---|---|---|---|
| False additional-source rate (2,700 singles) | 0.013 NP1 / 0.032 NP2 | 0.015 | **0.014** (worst morphology 0.089, TTPC2) | <= 0.02 | pooled yes, per morphology no |
| Resolvable-pair recall (870) | 0.334 / 0.505 | 0.310 | **0.416** | >= 0.80 | no |
| Simultaneous-pair recall (130) | 0.300 / 0.581 | 0.346 | **0.423** | >= 0.75 | no |
| Unequal-amplitude pair recall | 0.017 / 0.133 | 0.050 | 0.083 | - | - |
| Triple detection (>= 2 sources) | 0.15 / 0.08 (3 recovered) | 0.317 | **0.561** | - | - |
| Resolvable recall at matched 2% false-two (oracle, comparison only) | - | 0.463 | 0.570 | - | - |

B-P15: gradient-boosted classifier on line20 and ms3 statistics plus temporal and SNR features, per-recording
self-calibrated threshold (quantile 0.998 of development singles, chosen by nested LOMO; without nesting the
false-two rate rose to 3-5%). Real-data proxy on X1-X3 (300 sorter-isolated and 300 sorter-coincident events per
session): isolated two-source rate 0.0% for both P14-C and B-P15; coincident 2.3% (B-P15) vs 1.3% (P14-C); on X1,
P14-C flagged 16.3% of the coincident events as two sources vs 5.3% for B-P15. Coincident sorter events are not
verified overlaps.

### A. Localization and uncertainty (MEArec, 13 morphologies, LOMO, all conditions incl. pairs/triples, 4,320 events)

| Metric | P14 final (7 morph.) | ms3 (P14 estimator), same LOMO | **A-P15 ensemble (LOMO)** | Target | Reached |
|---|---|---|---|---|---|
| Median error (soma) | 16.0 / 17.8 um | 21.8 um | **15.7 um** (-28%) | - | - |
| p90 error | 57.0 / 45.5 um | 70.4 um | 69.8 um (-1%) | - | - |
| Disc coverage, complete population | 0.882 / 0.911 | 0.899 | **0.897** [0.887, 0.905] | >= 0.90 across conditions | no |
| Disc coverage, worst morphology | - | 0.832 | 0.793 | - | - |
| Disc coverage, singles / pairs / triples | - | - | 0.930 / 0.852 / 0.761 | - | - |
| Disc coverage, low SNR / dead pair / dt 1-3 ms | 0.739 (NP1 SNR<6) | - | 0.760 / 0.789 / 0.794 | - | - |
| Median disc area | 9,672 um^2 | 12,506 um^2 | 10,796 um^2 | - | - |

Line20 alone has the lowest median (14.1 um) but the widest tail (p90 83 um); the ensemble keeps most of its median
gain while matching ms3's tail. Coverage is calibrated overall and per SNR tertile (0.89-0.90) but not per condition:
the Mondrian bins (SNR x declared overlap) do not see undeclared overlaps, dead contacts or close-in-time pairs.

### F. Runtime (Apple Silicon laptop, numpy, 1 BLAS thread per process; `reports/phase15/stream15.json`)

| Metric | P14 (exp14_7) | **D-P15 batch path** | Target | Reached |
|---|---|---|---|---|
| Per-event compute latency, one event at a time (real IBL events, 3 sessions) | 2.28 ms median / 4.51 p95 / 9.71 p99 | **1.04-1.08 ms** median / 1.5-2.7 p95 / 1.9-4.9 p99 | <= 1 ms | no (4-8% over) |
| Per-event compute, batches of 32 | - | 0.76-0.84 ms per event (batch p99 27-34 ms) | - | - |
| Capacity, 2 workers (saturating producer, 3,000 events, no drops) | 773 events/s | **1,899-2,213 events/s** (D, L, O) | >= 1,500 | yes |
| Capacity, 1 / 4 workers (session D) | 428 / - | 802 / 3,076 events/s | - | - |
| Real-time replay, 2 workers (382 events/s offered) | - | 0 drops; end-to-end latency 4.1 ms median, 8.0 p95, 14-34 p99; max queue 22-45 | - | - |
| 3x real time, 2 workers (1,147 events/s) | - | 0 drops; 11.7 ms median, 67 ms p99; max queue 149 | - | - |

Caveats: the P14 numbers were measured on simulated snippets, each with its own geometry (worst case for batching:
the batch path measures 1.37-1.63 ms per event there, `batch_test.json`). The streamed worker path covers count and
position; identity association (0.02-0.03 ms per event) runs in the collector and is not in the measured path.
Memory was not re-measured in Phase 15. Overloaded replays (8,000-19,000 events/s offered) are kept in
`stream15_overload.json`: they drop events by design and are not capacity numbers. Best candidates for future
FPGA/ASIC acceleration: the per-grid-centre energy GEMM (K=1 map) and the Gram-identity pair search; both are
fixed-size multiply-accumulate kernels.

## Answers to the 10 questions

1. **Largest real improvements.** (a) Functional-cluster grouping for cross-animal decoding: spatial frequency
   +9.8 points on 6/6 new mice. (b) The amplitude-proportional assignment threshold for online identity: HQ coverage
   +16 to +23 points on new insertions. (c) The learned count model: resolvable-pair recall +11 points, triples +24
   points at an unchanged pooled false-two rate (cross-validated). (d) The batched runtime path: 2-worker capacity
   2.5-2.9x the Phase 14 figure.
2. **What failed and why.** Rate normalization, 10 or 50 ms bins, CORAL, ensembles, regularization sweeps, waveform-
   type groups, depth x type and depth x functional-cluster groups, and SNR- or quality-weighted pooling gave no
   cross-animal gain. Neuron-level structure that would carry orientation and position is not shared across mice,
   and group averages wash it out. Identity: low-SNR likelihood assignment, look-back, refractory, peak-distance and
   margin guards, and online splitting either cost precision or did nothing; online splitting also crashed on select.
   Overlap: without nested threshold selection the learned model's false-two rate rose to 3-5%, which is threshold
   overfitting. One extra decoding variant (e17, functional clusters + MLP) was stopped unfinished when the laptop
   ran out of memory.
3. **Difficult targets.** Orientation, Gabor position and natural-image identity across animals (all far from target);
   all-event identity coverage (10% vs 40%); resolvable and simultaneous overlap recall (42% vs 75-80%); coverage
   under every condition.
4. **Trade-offs.** Identity coverage vs precision is monotone on every split. Overlap recall vs false two-source rate
   is a strict frontier: at matched 2% the best recall is 57%. Median vs tail position error: line20 trades a lower
   median for a heavier tail. Latency vs throughput: batching lowers per-event cost (0.8 ms) but raises per-event
   latency to the batch time (up to 34 ms at p99).
5. **What generalizes.** Generalizes to new animals: the spatial-frequency gain (6/6 new mice), the identity coverage
   gain (3/3 new insertions, both window lengths) and the runtime gains (3 sessions). Does not generalize to new
   insertions: the IBL timing advantage. Generalization to new morphologies cannot be tested any more.
6. **Edge practicality.** The D-P15 batch path plus ID-P15 identity is practical on one laptop core per 800 events/s
   and two cores per 1,900-2,200 events/s, with about 4 ms median end-to-end latency at real-time rates. The
   cross-animal decoder is a linear model on 360 features. B-P15 adds 200 small trees (2,898 nodes, depth-limited) plus a second basis fit per
   event; it is a research configuration, not the deployment default.
7. **Exploratory.** Every biophysical result (B-P15, A-P15, uncertainty), because no unused morphology exists; the
   per-task routing; the real-data count proxy; and runtime measured on one machine.
8. **Experiments most likely to help.** Shared cross-animal latent models trained on many more mice (the learning
   curve is still rising); multitask decoding across stimulus blocks of the same mouse; a low-SNR identity stage that
   uses unit position and timing priors instead of waveform shape alone; a K=2 learned model with per-morphology
   calibration; condition-aware conformal bins (dead contacts, close-in-time pairs).
9. **Limiting resources.** Biophysical ground truth: 13 morphologies exist and all are used. The number of Allen
   mice with all stimulus blocks. A 16 GB laptop: 12 GB of swap use and the machine sleeping on battery stalled jobs
   for hours. No independent ground truth for real-neuron positions or overlaps.
10. **Next cycle.** Generate new biophysical morphologies (other layers, other cell databases) to restore an
    independent final set; scale cross-animal training (all Allen Visual Coding and Brain Observatory mice) with
    shared latent dynamics; add a low-SNR identity tier; compile the two hot kernels (numba or C) and re-measure
    single-event latency against the 1 ms target.

## Outstanding limitations

Six final mice give wide intervals; only spatial frequency survives a sign test. X4 could not be evaluated for
identity, real count or IBL decoding. Phase 15 does not change the frozen P14 deployment count model; B-P15 is
labelled a research configuration. Targets that were missed are reported as missed; none was redefined.
