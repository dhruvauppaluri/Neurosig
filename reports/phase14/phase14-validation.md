# Phase 14 validation: neural source identity, overlap, online localization and visual decoding

Software only; public data and simulation only. No SystemVerilog, FPGA, frozen-detector or fixed-point change.
Frozen configuration P14: `reports/phase14/frozen_p14.json`, hash `9d8006e97d8f`, written before any final data were
evaluated. Final sets were each used once (`scripts/phase14/run_final14.py`, `run_ibl14.py`, `run_vis14.py ... final`).
Registry: `reports/phase14/experiment_registry.jsonl`. Methods: `docs/phase14-localization.md`.

## 1. Baseline audit and reproduction

* Phase 12 and 13 reports, registries and code were reviewed. The Phase 12 P12-LR count model, the Phase 9-12
  space-time fitter and the Phase 13 IBL protocol were re-run unchanged inside the Phase 14 harnesses (identical
  events), so every Phase 14 number has a same-event baseline.
* **Discrepancy 1 (P12-LR).** Phase 12 reported 2.2% false two-source rate with 49.7% resolvable recall. Those figures
  come from a mixed pool where half the events used the line-source generator that matches the fitter. On purely
  biophysical, morphology-disjoint pools, the frozen P12-LR threshold gives a 3.0-5.2% false two-source rate with
  15.5-32% resolvable recall. At a matched 2% false two-source rate its recall is 9-26%.
* **Discrepancy 2 (Allen map).** The Phase 13 waveform-to-electrode defect was diagnosed: waveform rows follow the
  electrode's `local_index`, but the NWB electrodes table is not stored in that order. On one probe, 300 units' troughs
  sat a median 1,140 um from their peak channel under the old mapping and 0 um under the corrected one (71% exactly on
  the peak channel). The corrected caches put 76-77% of troughs exactly on the peak channel. See §7.

## 2. One complex neuron versus two neurons (14.2-14.3)

Morphology-disjoint MEArec/NEURON benchmark (`edge/phase14/bench14.py`). Development = TTPC1/LBC/MC, selection =
UTPC/STPC/NBC (plus a Neuropixels 2.0 geometry), final = 7 never-simulated cell models on NP1 and NP2 with background
from 4 fresh IBL insertions. Each pool has 900 singles in 6 conditions (clean, ±30% amplitude, artifact, 3 dead
contacts, correlated extra noise source, SNR 3-5), 480 pairs in 8 conditions and 60 triples.

Main development findings:

* **The phantom second neuron is mostly noise, not neuron shape.** With per-contact noise scaling, single-source F21
  stays near 2 even at SNR 3-6. Real background is spatially correlated. Whitening with the background's spatial
  covariance, estimated from event-free samples with no labels, raised select-morphology AUC from 0.72 (line source)
  and 0.82 (multi-scale) to 0.90-0.92. Spatio-temporal whitening was worse than spatial whitening alone.
* **A richer single-neuron null helps, up to a point.** Basis comparison at spatial whitening (select AUC): line20
  0.81; ms3 (g15, g40, axial dipole) 0.92; ms 0.90; msn 0.90; msw 0.89; ms5 0.87; msq 0.87. Requiring the second
  component to have its own negative trough removed most true pairs as well (rejected). A learned logistic
  discriminator overfit the development morphologies (rejected).
* **Thresholds transfer only with self-calibration.** A fixed F threshold moves with the recording's background. The
  deployable rule therefore calibrates the threshold per recording from development-morphology singles placed in that
  recording's own background.

**Identifiability limit (`exp14_3_identifiability.json`, select events re-rendered without noise).**

* With zero noise, one neuron's own complexity still produces a two-source energy gain (median 37, rising with SNR to
  76). Only 56% of resolvable pairs exceed the 98th percentile of that phantom gain (AUC 0.95). So even noise-free,
  model mismatch caps recall at a 2% false two-source rate near 56%.
* At real noise the noise-fitting gain (median 260) dominates: 28% of pairs exceed the single-source 98th percentile.
  At half the real noise, 48% do.
* With 32 contacts and this model family, resolvable-pair recall around 40% at about 2% false two-source rate is close
  to the measured limit. Further basis expansion was stopped.

**Final (unseen morphologies, fresh background; complete populations).** CI = Wilson 95%.

| Metric | P12-LR (frozen threshold) | P12-LR at matched 2% | **P14 deployable (self-calibrated 2%)** | P14 at matched 2% | P14 at matched 5% |
|---|---|---|---|---|---|
| NP1 false two-source rate | 3.3% | 2.0% | **1.3% [0.8, 2.3]** | 2.0% | 5.0% |
| NP1 resolvable-pair recall | 21.6% | 18.6% | **33.4% [28.3, 39.0]** | 42.9% [37.4, 48.6] | 57.8% |
| NP1 simultaneous resolvable pairs | 20.0% | 20.0% | **30.0%** (n = 40) | 45.0% | 62.5% |
| NP1 all pairs | 15.0% | 12.5% | **23.3%** | 30.6% | 41.9% |
| NP2 false two-source rate | 3.0% | 2.0% | **3.2% [2.3, 4.6]** | 2.0% | 5.0% |
| NP2 resolvable-pair recall | 31.6% | 25.6% | **50.5% [44.9, 56.1]** | 38.2% | 56.5% |
| NP2 simultaneous resolvable pairs | 34.9% | 32.6% | **58.1%** (n = 43) | 48.8% | 67.4% |
| Separation AUC, singles vs resolvable pairs (NP1 / NP2) | 0.75 / 0.79 | | **0.90 / 0.90** | | |

| P14 final, further metrics | NP1 | NP2 |
|---|---|---|
| False two-source rate by condition (clean / amplitude / artifact / dead / correlated / low SNR) | 2.7 / 2.7 / 0.7 / 1.3 / 0.7 / 0% | 4.7 / 4.0 / 2.0 / 3.3 / 5.3 / 0% |
| Pair recall by condition (dt 0 / 1-3 / 4-9 / 10-20 / near 20-60 um / unequal / same shape / dead) | 23 / 35 / 28 / 35 / 20 / 2 / 22 / 22% | 45 / 53 / 38 / 50 / 37 / 13 / 42 / 27% |
| Both sources recovered within 60 um (resolvable pairs) | 30.4% | 41.5% |
| Median error, correctly recovered pairs | 15.5 um | 18.6 um |
| Triples: all three recovered (P12: 0%) | 15.0% | 8.3% |
| False three-source rate (pairs / singles) | 0.6% / 0% | 0.8% / 0% |
| Single-source primary error median / p90 | 16.0 / 57.0 um | 17.8 / 45.5 um |
| Hard count error (singles / multi-source) | 1.3% / 74.8% | 3.2% / 61.1% |

Answers:

* NeuroSig now distinguishes one complex neuron from two at an operating point that keeps false additional neurons at
  1.3-3.2% on independent morphologies. It recovers a third to a half of resolvable pairs, about 2x P12-LR at matched
  false-positive rates.
* The targets of ≥80% resolvable recall, ≥60% biophysical recall and ≥65% simultaneous recall are **not met**. Neither
  is the ≤10 um pair error. Unequal-amplitude pairs (2-13%) and near pairs (20-37%) are the hardest.
* Real recordings (`exp14_3_real_*.json`; the sorter is not ground truth):
  * Isolated real events are called two in 0-0.25% of cases, sorter-coincident events in 0.9-3.2%. P11 did not
    separate these groups (76-85% two), and P12-LR barely did (4.0 vs 4.1%).
  * Pooled templates of good sorter units (N = 50 spikes) are called two in 67-73% of dev/select units and 12.5% of
    final units. At high effective SNR, real single neurons still exceed the null model. The count rule is therefore a
    single-event rule and must not be applied to pooled waveforms.
* Operational default retained: **K=1 plus a calibrated overlap probability.** K=2/3 outputs are flagged and carry the
  probability.

## 3. Online neural identity without sorter labels (14.4)

Detector-v2.0 events (frozen detector), sorter labels used only for evaluation. Frozen OU-P14 (energy gain ≥ 80,
SNR ≥ 6 to assign, no prototype merging), selected on dev A-C only.

| Split | Precision (all matched) | **Precision, high-quality subset** | **Coverage, high-quality subset** | Coverage, all detector events | Merge rate | Fragmentation | Adjusted Rand index | **Unit vs event position spread** |
|---|---|---|---|---|---|---|---|---|
| dev v0 (Phase 12-style thresholds) | 58.0% | 90.3% | 62.3% | 57.1% | 8% | 1.42 | 0.53 | -73% |
| dev, frozen | 92.1% | 99.7% | 64.6% | 13.7% | 8% | 1.00 | 0.78 | -59% |
| select D-S (16 sessions) | 84.3% | 94.9% | 45.5% | 11.2% | 13% | 1.03 | 0.75 | -69% |
| **final T-W (4 new insertions)** | 74.4% | **96.7%** | **59.5%** | 5.7% | 45% | 1.06 | 0.59 | **-61%** |

High-quality subset = events matched to sorter units with label 1 and SNR ≥ 8, predefined.

* Precision on the high-quality subset meets the ≥95% target on dev and final (94.9% on select). Its coverage meets
  ≥50% on dev and final (45.5% on select).
* Unit-level position variability is 59-73% below single-event variability (target ≥ 50%).
* Over all matched events, precision is 74-92%. Coverage of all detector events is low (6-14%) because events below
  SNR 6 are never assigned; this is reported, not hidden. Per-SNR results are in `exp14_4_*.json`.
* Weak sessions (N, S) produce no units rather than wrong ones. Two of four final sessions had high merge rates (sorter
  units sharing one online unit): 0.40-0.67 of units with ≥5 matched events.
* Initialization takes a median 32-62 events per unit. Most low-SNR sorter units never initialize (71-93% of units
  with ≥20 events).
* Runtime: 0.03-0.6 ms per event; memory a few KiB per unit.

## 4. Drift-aware unit localization (14.5)

Ground-truth streams (`exp14_5_*.json`): 5 neurons, real background noise, probe drift linear 30 um, step 25 um, sine
20 um, 120 s; 3 seeds each.

| Profile | Static pooled error, final | **Kalman (frozen), final** | Kalman with online identities, final | Identity precision (true labels) | Fragmentation | Assigned in last quarter |
|---|---|---|---|---|---|---|
| none | 0.4 um | 0.5 um | 0.5 um | 100% | 1.0 | 80% |
| linear 30 um | 9.8 | **4.0** | 4.8 | 100% | 1.4 | 55% |
| step 25 um | 8.7 | **2.1** | 1.5 | 99.9% | 1.75 | 46% |
| sine 20 um | 13.9 | **5.7** | 5.1 | 100% | 1.8 | 56% |

Errors are axial tracking errors after removing each unit's constant localization bias.

* Tracking error falls 2.4-4x against a static pooled position (target met). Selection morphologies give the same
  ordering (Kalman 4.5 / 1.5 / 4.0 um vs static 10.0 / 12.1 / 10.7).
* Identity is never mixed (precision ≥ 99.9%). Units **fragment** under drift (1.4-1.8 online units per neuron) and
  later events go unassigned.
* Relinking a new unit to a silent old one cut simulated fragmentation from 1.37-1.57 to 1.10-1.32. It also merged
  distinct real units across the gaps between 2 s windows (dev precision 92% → 86%), so it was rejected for real data.
* Position uncertainty (Kalman variance) and drift (innovation / change point) are reported separately by the tracker.
  Probe displacement and neural movement are not distinguishable from one neuron alone. The common-shift assumption
  is a simulation assumption.

## 5. Uncertainty over the complete population (14.6)

| | Primary-source coverage (target 90%) | Median area | Unbounded | Calibrated on |
|---|---|---|---|---|
| Phase 12, complete population | 85.7% | n/a | 53% | p12 pools |
| P14 conformal disc, select / select NP2 | 85.2% / 82.8% | 9,672 um^2 | 0% | dev morphologies |
| **P14 conformal disc, final NP1 / NP2** | **88.2% [86.4, 89.8] / 91.1% [89.5, 92.5]** | 9,672 um^2 | 0% | dev morphologies |
| P14 profile-likelihood region, final NP1 / NP2 | 89.8% / 82.1% | 6,150 / 5,500 um^2 | 88% | dev |

* Disc coverage by SNR, final NP1: 74 / 87 / 91 / 93% for SNR <6 / 6-10 / 10-15 / ≥15. By kind: singles 92%, pairs
  84%, triples 62%. Low-SNR and multi-source events under-cover; this is reported, not corrected after the fact.
* Count uncertainty: the overlap probability has ECE 0.04 on both final geometries (balanced prior).
  "Confidently incorrect" count decisions (confidence ≥ 0.9 and wrong) are 0.6% and 0.8% of all events (target ≤ 5%).
  Only 8-12% of events reach that confidence: the probability is honest but rarely decisive.
* Position error under model mismatch is dominated by a morphology-dependent bias (current centroid vs soma). Likelihood
  shape therefore adds little, profile regions are almost always unbounded, and calibration shifts by 3-7 points across
  morphologies.
* Identity uncertainty is reported as the abstention rate (ambiguous and gray-zone events) and the high-quality vs
  all-event precision gap in §3.

## 6. Runtime (14.7)

* Fast exact implementation (`edge/phase14/fast14.py`, equivalence-tested): the full K≤3 ms3 analysis went from 26.6
  ms to 3.2 ms per event.
* The frozen reduced search (2 first-centre candidates, 1 refinement round) takes **1.6-2.2 ms median** for the
  count path, including the K=1 position, across select and final pools. It costs 1.7 points of resolvable recall on
  select against the full search.
* The lazy fine grid removed the dead-contact cache rebuilds: p95 fell from 27-33 ms to 7.6 ms, with identical outputs.
* Throughput and worker scaling (`exp14_7_runtime.json`, idle machine, frozen configuration):
  * Single process: median latency 2.28 ms, p95 4.5 ms, p99 9.7 ms; peak traced memory 2.1 MiB.
  * Measured detector event rates are 232 events/s median and 461 events/s maximum across 23 IBL sessions.

  | Workers | Events/s | Margin vs median rate | Margin vs maximum rate | Latency p99 |
  |---|---|---|---|---|
  | 1 | 428 | 1.85x | 0.93x | 5.1 ms |
  | 2 | 773 | 3.3x | **1.68x** | 10.3 ms |
  | 4 | 973 | 4.2x | 2.1x | 14.7 ms |

  Every event is fully processed. Online association adds 0.03-0.6 ms per event.
* The ≤ 2 ms median target is met on the count path under load (1.6-2.0 ms). It is missed slightly on the whole
  per-event path at 2.28 ms: the uncertainty map is included in that figure.

## 7. Repaired Allen evaluation (Section 8 of the brief)

* Corrected caches: the median unit has its waveform trough on its peak channel (73-77% exactly; Phase 13: median
  86 um off). `edge/phase14/allen_map.py` holds the mapping, with regression tests for permuted tables and two probes.
* Corrected comparison `dmap_p12c` (P12 line20 position of the corrected mean waveform) against `dmap_peak`
  (peak-channel position): differences of -0.8 to +0.4 points in every task and split, all n.s. (final: 2-4 of 6
  mice positive). The corrected positions sit a median ~5 um from the peak channel, so the two maps are nearly the
  same features. Dev/select Allen mice had been inspected in Phase 13; only the 6 final mice are an untouched
  evaluation of this comparison.

## 8. Visual decoding (Sections 9-12 of the brief)

Allen: 2 dev, 9 select and 6 final mice, every session a different mouse, 0-250 ms windows, blocked 5-fold CV,
balanced accuracy. Deltas are means across mice, with the number of mice positive and the session-level Wilcoxon p.

**Temporal codes (Section 9).** Final mice (n = 6), with select (n = 9) in parentheses:

| Task (chance) | Counts | 50 ms bins | Δ 50 ms bins vs counts | Δ count + latency vs counts | Δ matched MLP, 25 ms bins vs counts |
|---|---|---|---|---|---|
| SG orientation (0.167) | 0.712 | 0.759 | **+4.6, 6/6, p = 0.031** (+5.3, 9/9) | +3.0, 6/6 | +2.6, 6/6 |
| SG spatial frequency (0.20) | 0.881 | 0.921 | **+4.0, 6/6, p = 0.031** (+3.8, 9/9) | +2.2, 6/6 | +2.7, 6/6 |
| Gabor vertical position (0.111) | 0.597 | 0.630 | **+3.3, 6/6, p = 0.031** (+3.3, 8/9) | -0.7 | -4.0 |
| Gabor horizontal position (0.111) | 0.455 | 0.474 | +1.8, 6/6 (+1.5, 8/9) | -0.7 | -3.5 |
| DG orientation (0.125) | 0.689 | 0.691 | +0.3, 2/6 (-0.1) | +1.7, 5/6 (+1.3, 7/9) | -24.5 |
| NS image identity (0.008) | 0.800 | 0.794 | -0.6 (+0.4) | n/a | n/a |

* **The ≥ 3-point temporal advantage replicates** on independent mice:
  * Allen static-grating orientation, spatial frequency and Gabor vertical position (6/6 final mice).
  * IBL contralateral contrast on the 4 fresh insertions T-W (+3.3 pts, 3/4 sessions, CI +0.1 to +6.6; Phase 13
    protocol unchanged, `ibl14_final.json`).
* It does not hold for drifting gratings (sustained response), natural images, or IBL stimulus side (-0.9 pts).
* Which resolutions help:
  * 50 ms bins capture the gain; 25 ms bins add nothing more for units (and cost 2.5-3.5 pts on natural images).
  * At area level, 5 ms bins beat 50 ms only for spatial frequency (+2.5, 6/6).
  * A two-number-per-unit latency code (count + spike-time centroid) recovers 50-75% of the gain for gratings with
    40% of the parameters.
  * The matched MLP gains on static gratings but collapses on drifting gratings: capacity is not the explanation.
* IBL stimulus presence remains at chance (0.511 vs 0.517). IBL choice is decodable (0.555-0.559), so side decoding
  stays partly choice-confounded.

**Localization and decoding (Section 11).** Final mice; 50 ms counts are the reference:

| Comparison | SG orientation | SG spatial frequency | DG orientation | Natural images | Gabor x | Gabor y |
|---|---|---|---|---|---|---|
| Depth map (peak channel) vs counts | -11.5 | -12.0 | -7.4 | -25.9 | -6.4 | -9.0 |
| Corrected-position map vs shuffled positions | -0.3 | -1.0 | +0.8 | -0.5 | +0.5 | +0.6 |
| Corrected-position map vs random coordinates | -1.4 | -1.3 | -0.3 | **-2.3 (0/6)** | +1.0 | -0.7 |
| Corrected-position map vs dimension-matched random projection | -1.2 | -1.7 | -0.7 | -3.1 | 0.0 | -0.3 |

* Spatial maps built from where units sit on the probe never beat spike counts, shuffled positions, random coordinates
  or a dimension-matched random pooling, in any task including stimulus position. This holds with corrected positions
  and on 6 untouched mice.
* CCF cortical-surface maps (select, 7 mice) lose 11-52 points to counts.
* Depth along a probe is not the retinotopic axis: Gabor position is decodable from unit identities (0.45-0.60 vs
  0.11 chance), but not better from their depths.
* IBL T-W: a sorter-depth map beat counts for stimulus side by +2.2 (4/4, CI +1.4 to +3.6) and beat shuffled depths
  by +5.9 (4/4). It did not beat a dimension-matched random projection (+1.4, n.s.), and per-spike depth did not beat
  unit depth (-2.9).
* Conclusion: within a recording, location information is redundant with unit identity. The tested codes are not
  organized along the sampled axis at a scale the probe resolves. The Phase 13 simulation explains when localization
  would matter.

**Cross-animal generalization (Section 10).** Decoders trained on other mice and tested on each held-out mouse,
using per-area (VISp, VISl, VISal, VISrl, VISam, VISpm) × 25 ms population features:

| Task (chance) | Final mice (n = 6), session z-scored | Per mouse | Area identities permuted | Shuffled training labels | Select leave-one-out (n = 9) |
|---|---|---|---|---|---|
| SG spatial frequency (0.200) | **0.422** | 0.37-0.51 (6/6 > chance) | 0.302 | 0.200 | 0.416, 9/9 |
| Gabor horizontal position (0.111) | **0.191** | 0.10-0.26 (5/6) | 0.127 | 0.112 | 0.167, 8/9 |
| Gabor vertical position (0.111) | **0.176** | 0.13-0.22 (6/6) | 0.142 | 0.110 | 0.218, 9/9 |
| SG orientation (0.167) | **0.232** | 0.21-0.26 (6/6) | 0.178 | 0.157 | 0.203, 6/9 |
| DG orientation (0.125) | **0.195** | 0.15-0.23 (6/6) | 0.160 | 0.135 | 0.165, 6/9 |
| NS image identity (0.008) | **0.044** | 0.029-0.068 (6/6) | 0.017 | 0.008 | 0.043, 9/9 |

* **Meaningful above-chance cross-animal decoding is achieved**, with per-area population features and label-free
  per-session normalization. Spatial frequency reaches 2.1x chance in every held-out mouse, and natural-image identity
  5x chance.
* The gain comes from anatomically grounded regional aggregation: permuting area identities removes most of it. It is
  far below within-animal decoding (0.92 / 0.80).
* Phase 13's depth-map features were near chance across mice. No neuron identities are shared across sessions here.

**Natural images and unseen stimuli (Section 12).**

* The Allen NWB files carry no stimulus images, so image-feature prediction and reconstruction were not attempted.
  The only image claim is identity classification among 118 known images (within mouse 0.80; across mice 0.044, 5x
  chance).
* Parameterized reconstruction for held-out static gratings: (orientation, spatial frequency) combinations never seen
  in training, a third of all combinations. Final mice:
  * Orientation median error 18.5-19.0 deg (chance 45 deg; select 17.1-19.4).
  * log spatial frequency R² = 0.78-0.81 (select 0.75-0.79).
  * Area-level features are much worse (46 deg, R² 0.50).
* This is reconstruction of stimulus parameters, not of images or of subjective experience.

## 9. Targets

| Objective | Historical baseline | Updated baseline (same events) | Best development | **Locked evaluation** | Target met |
|---|---|---|---|---|---|
| False additional-neuron rate | P11 75-85%; P12-LR 2.2% (mixed pool) | P12-LR 3.0-5.2% (biophysical) | ms3 sp, 1.7% dev | **1.3% NP1 / 3.2% NP2** | yes (≤ 5%) |
| Single-source acceptance | - | - | 98.3% | **98.7% / 96.8%** | yes (≥ 90%) |
| Resolvable-pair recall | P12-LR 49.7% (mixed) / 13.9% biophysical | P12-LR 18.6-25.6% at matched 2% | 38-42% select | **33.4% / 50.5%** (matched 2%: 42.9 / 38.2%) | no (≥ 80%) |
| Simultaneous-pair recall | 35.1% | 20-33% | 29-49% | **30.0% / 58.1%** | no (≥ 65%) |
| Pair error, recovered pairs | 4.0 um (line-source) | - | 16-23 um | **15.5 / 18.6 um** | no (≤ 10 um) |
| Three-source recovery | 0% | 0% | 3-10% | **15.0% / 8.3%** | improved |
| Online precision, high-quality subset | Phase 12 purity Q 0.26 / R 0.93 | v0 90.3% | 99.7% | **96.7%** | yes (≥ 95%) |
| Online coverage, high-quality subset | - | 62.3% | 64.6% | **59.5%** | yes (≥ 50%) |
| Unit vs event position variability | -61% at N = 5 (sorter-assisted) | - | -59% | **-61%** (sorter-free) | yes (≥ 50%) |
| Drift tracking error vs static | - | static 8.7-13.9 um | Kalman | **2.1-5.7 um** | yes |
| Primary coverage, complete population | 85.7% (53% unbounded) | - | 90.0% dev | **88.2% / 91.1%** (0% unbounded, 9,672 um^2) | about 90% (NP1 −1.8) |
| Confidently wrong count decisions | - | - | 1.2% | **0.6% / 0.8%** | yes (≤ 5%) |
| Multi-source path median latency | 5.65 ms | - | 1.6-1.9 ms | **1.96-2.20 ms (count); 2.28 ms total** | about met |
| Throughput above event rate | 2 workers 318 ev/s | - | - | **2 workers 773 ev/s = 1.68x max rate** | yes |
| Temporal advantage ≥ 3 pts | +3-5 (Phase 13) | - | +3.3 to +5.3 (select) | **+3.3 to +4.6, 6/6 mice; IBL contrast +3.3** | yes (SG, SF, Gabor y, IBL contrast) |
| Localization beats activity alone | no (Phase 13) | corrected map = peak map | none | **none** | no (redundant) |
| Above-chance cross-animal decoding | near chance | - | 9/9 mice (SF) | **6/6 mice, SF 2.1x chance** | yes |

## 10. Answers

1. **One complex neuron vs two?** Partly.
   * At a self-calibrated operating point, P14 keeps false additional neurons at 1.3-3.2% on 7 unseen morphologies
     and recovers 33-51% of resolvable pairs (AUC 0.90).
   * Even without noise, single-neuron complexity limits recall at 2% false positives to about 56%.
   * Pooled real templates at high SNR still exceed the null, so the rule is for single events only.
2. **False additional-neuron rate across independent biophysical models:** 1.3% [0.8, 2.3] (NP1) and 3.2% [2.3, 4.6]
   (NP2) on unseen morphologies. By condition 0-5.3%, highest with correlated extra noise on NP2.
3. **Overlap recovery at acceptable false positives:** 43% of resolvable pairs at 2% and 58% at 5% (NP1, matched), and
   38% / 57% on NP2. Simultaneous pairs 45-62%. Unequal-amplitude pairs are rarely recovered (2-13%).
4. **Online identification without sorter labels:** yes for high-quality events.
   * 96.7% precision and 59.5% coverage on fresh sessions; unit positions 61% less variable than single events.
   * Not for weak events: those below SNR 6 are abstained on.
   * Two of four fresh sessions show unit merges.
5. **Drift:** a Kalman tracker follows simulated probe drift with 2-6 um error (static pooling 9-14 um). Identity
   never mixes, but units fragment (1.4-1.8x), and abrupt drift is the hardest.
6. **Calibration:**
   * Count probability is calibrated (ECE 0.04; confidently wrong ≤ 0.8%).
   * Position discs reach 88-91% complete-population coverage but are large (r ≈ 55 um), because model bias dominates.
     Low-SNR (74%) and triple (62%) events under-cover.
   * Identity uncertainty is reported through abstention and the precision gap, not as a calibrated probability.
7. **Event rate:** yes. 2.3 ms per event; 2 worker processes sustain 773 events/s, 1.68x the busiest measured session.
8. **Stimulus features across independent recordings:** yes.
   * Spatial frequency (0.42 vs 0.20), grating orientation, Gabor position and natural-image identity (5x chance)
     decode above chance in every held-out mouse.
   * The features are per-area population activity with per-session normalization.
9. **Which temporal features help:** 50 ms response-time structure in the first 250 ms adds 3-5 points for
   static-grating orientation and spatial frequency, Gabor vertical position and IBL contrast. Spike-latency centroids
   carry most of it. Finer bins add nothing except area-level 5 ms for spatial frequency. Sustained drifting-grating
   and natural-image responses gain nothing.
10. **Does localization add information beyond activity?** No, in any task or dataset tested, including stimulus
    position. Corrected unit positions equal peak-channel positions, and depth maps lose to counts and match random
    coordinates.
11. **Unseen animals and stimuli:**
    * Unseen animals: yes at the population/area level, far below within-animal accuracy.
    * Unseen stimulus-parameter combinations: orientation 19 deg median error, log-SF R² 0.8.
    * Unseen natural images: not tested, because the images are not in the data files.
12. **Claims still without independent ground truth:**
    * Real-neuron positions (sorter depths are proxies; MEArec bias is 16-27 um).
    * Real-data source counts (sorter coincidence is a proxy).
    * Online identity precision (sorter labels are not ground truth).
    * Drift behaviour of real neurons versus the simulated common shift.
    * Biophysical realism: 13 BBP L5 cell models, NP1 noise reused for NP2.
    * Unit-level counts on pooled templates.
    * Any image reconstruction.

## 11. Remaining limits and recommendation

* Only 13 cell models exist in MEArec's BBP set. All 7 unseen ones are now used; a future final needs new morphologies.
* Real IBL windows are 2 s long and discontinuous, so real drift and long-range identity could not be tested.
* Recommendation:
  * Keep P14 single-event K=1 plus overlap probability, with OU-P14 unit association and Kalman unit positions, as the
    edge output.
  * Spend future effort on hybrid ground truth (injected real units) and longer contiguous recordings rather than
    richer spatial models.
  * For vision, use timing-rich, area-level population features. Localization is not worth hardware cost for these
    decoding tasks.
