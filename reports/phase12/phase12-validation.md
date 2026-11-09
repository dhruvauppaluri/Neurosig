# Phase 12 validation: unit-level localization, spatiotemporal fitting, source counting, uncertainty, runtime

Software only. Frozen configuration **P12** (hash `5f07b9153c22`, `edge/phase12/configs.py`) was decided on dev,
checked on select and evaluated once on the fresh final sessions **Q-S** and the **p12final** pool
(`reports/phase12/final12.json`). All numbers below are read from the JSON reports in this folder; the experiment
registry (`experiment_registry.jsonl`, 33 entries) lists every hypothesis, including rejected ones.

**Proxies, not truth.** Real-neuron positions are not independently known. On real data, "stability" means agreement
between estimates of the same sorter unit, which is a sorter-derived proxy. Stability is not accuracy. Accuracy is
measured only on simulations, where the source position is known.

## 0. Baseline reproduction

`baseline_repro.json`: the Phase 11 11A dev matched-event spreads (weighted_scan A/B/C 14.4 / 14.4 / 15.7 um) and the
11E OV-A select metrics (resolvable recall 94.1%, false two 4.2%, simultaneous 73.9%, triples 0%) reproduce exactly.
Runtimes differ because they depend on machine load.

## 1. Unit-level localization (12.2)

Disjoint pools of N spikes per sorter unit; stability is the median pairwise distance between estimates from disjoint
pools (N = 1 is the matched event-level baseline). Estimator: `l20_robust` (Huber-robust mean of E3 trough
amplitudes, line20), chosen on dev.

| Split (matched events) | N = 1 | N = 5 | N = 10 | N = 25 | N = 50 |
|---|---|---|---|---|---|
| dev A-C (units) | 24.7 um (103) | 9.2 (-63%) | 6.5 (-71%) | 3.6 (-81%) | 4.2 (-78%, 14 units) |
| select D-P | 17.3 (427) | 7.1 (-59%) | 4.8 (-70%) | 2.9 (-81%) | 1.4 (-88%, 47) |
| **final Q-S** | 17.8 (87) | 6.9 (**-61%**, CI -66/-55) | 4.4 (-72%) | 3.3 (-80%) | 1.6 (-90%, CI -93/-83, 17) |

* The 25% variability-reduction target is met from N = 5 on every split, with unit-bootstrap intervals that exclude 0.
  Including every sorter event (low-SNR events and events with no detector partner), final drops from 32.5 um to 10.6 /
  4.2 / 3.3 um at N = 5 / 25 / 50.
* **Plateau.** Random-pool disagreement levels off at about 2-4 um by N = 25. But **pools taken early and late in a
  session disagree 2-7 times more than random pools** (final mean-estimator ratio 2.2 at N = 5 rising to 6.6 at N =
  50). The positions drift or are non-stationary across the hour-long session, so the plateau measures precision
  within a period, not one fixed position. The drift is reported, not averaged away.
* Mean ~ robust > joint > SNR-weighted position aggregation. The joint per-spike-amplitude fit is not better.
* **Online cost.** Time to collect N detector-confirmed spikes of one unit (median unit rate 2.7 Hz): median 1.9 s
  for N = 5, 9.3 s for N = 25, 18.6 s for N = 50 (90th percentile 27 s and 55 s). Each update costs one mean and one
  line20 fit (< 1 ms).
* **Sorter-free online grouping** (rho 0.9, the only threshold with dev purity >= 0.9): dev assigns 6.6% of detector
  events at 94.6% purity; select 3.4% at 93.6%. On final, R gives 15.2% at 93.4% and **Q only 25.9% purity**; S has no
  groups. **Unit-level pooling that is genuinely online and sorter-free is not reliable yet.** The pooling gains above
  are offline and sorter-assisted.
* Session S is a weak recording: only 6% of its sorter events have a detector partner, and no unit reached 10
  matched events. It is reported, not dropped.

## 2. Single-event spatiotemporal variants (12.3)

The same matched events and units are used throughout. Changes are relative to the Phase 11 headline engine (ws_A:
weighted_scan on the sorter-time snippet).

| Variant | dev median / p90 | select median / p90 | final median (CI) / p90 | biophysical median error, SNR 8 (select / final) |
|---|---|---|---|---|
| ws_A (baseline) | 14.4 / 66.3 um | 11.2 / 50.0 | 12.5 / 30.2 | 66.6 / - |
| ws_E3 | -4% / -21% | +1% / -19% | -4% / +1% | 70.6 / 67.7 |
| mean3 (adopted) | -10% / -21% | -6% / -19% | **-13% (-26, +3) / +9%** | 18.5 / 22.1 |
| mean3_hp | -17% / -12% | -13% / -24% | -19% (-26, -4) / -3% | 22.8 / - |
| matched filter | -3% / -16% | -12% / -9% | -33% / +15% | 17.5 / - |
| rank-1, GLS, contact selection, free width | +30% to +70% / up to +146% | worse | worse | best bias (rank1_gls 12.0 / 17.3) |
| st_vp (K=1 space-time) | +149% | +245% | - | 18.0 / 18.8 |

* **Targets missed** (-15% median, -20% p90, on real data and independent simulations). On dev and select, E3
  alignment with line20 cuts the p90 tail by about 20%, but the final sessions start with a small tail (p90 30 um)
  and show no gain. Coverage stays >= 99.7%.
* **Stability and accuracy disagree.** Contact selection, spatial-noise GLS and rank-1 denoising reduce biophysical
  bias by 4-8 um, yet they destabilize real events. High-pass filtering stabilizes real events but worsens low-SNR
  biophysical accuracy.
* **The Phase 11 headline engine (weighted_scan) is biased by 45-58 um on model-mismatched biophysical sources, even
  noise-free.** The line20 family is biased by 13-20 um. Its real-data stability does not indicate accuracy, and P12
  replaces it with mean3 + line20.
* **Identifiability.** The line20 profile-likelihood area predicts real-event deviation: Spearman 0.40 (dev), 0.52
  (select), 0.49 (final). Median deviation by area quartile on final is 7.1 / 9.7 / 17.1 / 35.6 um. Ambiguity is
  therefore carried in the uncertainty output (section 4).

## 3. Overlap and source counting (12.4)

The expanded benchmark per pool has 2,240 singles: 240 line-source singles and 2,000 biophysical singles across four
conditions (clean, amplitude jitter, artifact, 3 dead contacts) with real noise. It also has 400 biophysical pairs over
dt bins, 600 line-source pairs, and 340 triples.

| Final pool p12final | P11 (Phase 11 rule) | P12-LR (adopted) |
|---|---|---|
| False two, line-source singles | 5.4% [3.2, 9.0] | 0.4% [0.1, 2.3] |
| **False two, biophysical singles** | **83.5% [81.8, 85.0]** | 2.5% [1.9, 3.2] |
| False two, all singles | 75.1% | **2.2% [1.7, 2.9]** |
| Resolvable-pair recall (all / line-source / biophysical) | 77.0% / 94.0% / 60.9% | 49.7% / 87.2% / 13.9% |
| Simultaneous resolvable pairs | 64.0% | 35.1% |
| All pairs | 56.7% | 30.0% |
| Triples (all) | 0% | 0% (K = 3 off) |
| Median error, recovered resolvable pairs | 5.9 um | 4.0 um |

* **Main finding.** A line source with a free waveform cannot represent one extended neuron. P11 declares two
  sources for **67% of noise-free** MEArec singles, 76-84% of noisy biophysical singles, and **85% of real
  detector-confirmed events** (dev). On real select and final events, P11's two-source rate is the same on
  sorter-isolated and sorter-coincident events (84% vs 85%; 76% vs 81%). **The Phase 9-11 false-two figures (4-9%)
  were artifacts of the line-source benchmark.**
* P12-LR (logistic on normalized drops, temporal rank, trough count, timing spread, chi2, and second-source
  polarity, correlation, energy and separation; threshold set to 2% dev false two) keeps false two at or below 5%
  with its interval (final 2.2%, upper bound 2.9%; select 4.8%, upper bound 5.8%). The cost is recall: **biophysical
  resolvable-pair recall falls to 14-23%**. On real events it declares two on 3.9-4.0% of isolated events, but it
  barely separates coincident from isolated events (AUC 0.60 select, 0.59 final).
* An empirical waveform dictionary as the single-neuron null was rejected. With the event's own template left out,
  singles produce as much second-source evidence as true pairs (dev median F 228 vs 210), and recall at 2% false two
  is 0.
* Targets: false two <= 5% with interval **met (P12-LR)**. Resolvable recall >= 90%, simultaneous >= 85% and triples
  >= 40% **not met**. Pair error <= 10 um met for recovered pairs (4.0 um), but on few pairs.
* **Identifiability conclusion.** With a 32-contact Neuropixels section and these model families, a second neuron is
  not reliably distinguishable from the extended footprint of the first under realistic waveforms. K = 1 with an
  overlap probability should be the default, and K = 2 an experimental, low-recall output.

## 4. Uncertainty (12.5)

Complete-population scoring: every event is in the denominator. A primary estimate with no true source within
60 um, or no estimate, counts as uncovered. Count/identity uncertainty is scored separately.

| p12final | Primary coverage, all events | True-source coverage | Region size |
|---|---|---|---|
| Gaussian chi2(2) 90% | 43.4% | 35.3% | 342 um^2 |
| Phase 11 protocol (Mondrian over matched primaries) | **81.6%** [80.4, 82.8] (nominal 90%) | 62.1% | 5,905 um^2 |
| P12 Mondrian over all events | 85.7% [84.5, 86.7] | 64.9% | 53% of regions unbounded |
| P12, top 80% accepted by region size | 90.5% (94.2% of primaries matched) | - | - |

* **The Phase 11 90.5% coverage claim does not hold for the complete population.** About 11% of events have a
  primary that is not any true source (count or identity errors). No bounded region can reach 90%, so P12 reports
  such groups as unbounded instead of hiding them.
* Count uncertainty: Brier 0.227. 11% of events are confidently wrong (P >= 0.9 or <= 0.1 on the wrong side).
* **Real single-event uncertainty (proxy).** A disc sized by the identifiability area (c per SNR group, fitted on
  dev) covers the leave-one-out unit median for 90.9% [90.0, 91.8] of final events with a median area of 2,993 um^2.
  A fixed disc fitted the same way covers 96.4% with 8,690 um^2. The identifiability disc is calibrated, about 3x
  smaller, and more uniform across sessions. The reference is a unit-level proxy, not truth.

## 5. Runtime (12.6)

Real detector-v2.0 streams (select 40 windows, final 24 windows), one process per core, this Apple Silicon machine:

| | select (258 ev/s measured) | final (206 ev/s) |
|---|---|---|
| Single-source path (E3 + mean3 + FastLine20 + area), median | 0.88 ms | 0.90 ms |
| Full P12 event (plus fast K<=2 analysis and count rule), median / p95 | 5.57 / 9.54 ms | 5.65 / 9.95 ms |
| One core sustained | 162 ev/s (margin 0.63) | 158 ev/s (0.77) |
| 2 workers / 4 workers | 321 / 612 ev/s (1.25x / 2.4x) | 318 / 595 ev/s (1.5x / 2.9x) |
| Peak traced memory, one 2 s window | 89 MiB | 89 MiB |

* FastLine20 matches the reference line20 fit within 0.27 um median (0.8 um p95), 8x faster. The analytic-Jacobian
  K<=2 refine with the closed-form shared eigenvalue keeps 99.75% of count decisions identical and cuts the analysis
  from 8.5 to 5.2 ms.
* The <= 1 ms single-source target is met. The <= 2 ms multi-source target is **not met**: the shared-waveform search
  and two variable-projection refinements remain. The cheap gate was rejected because it still invokes the K<=2
  analysis on 93% of events. **Sustained throughput needs >= 2 cores.** A single-core FIFO queue drops 39-44% of
  events. Difficult events are never silently skipped.

## Targets

| Target | Baseline | Result (final) | Reached |
|---|---|---|---|
| Unit-level variability -25% | 17.8 um (N = 1) | -61% at N = 5, -90% at N = 50 | yes (offline, sorter-assisted; drift present) |
| Plateau <= 50 spikes | - | ~2-4 um by N = 25 (random pools) | yes, with early/late drift 2-7x |
| Single-event median -15% | 12.5 um | -13% (CI -26, +3) | no |
| Single-event p90 -20% | 30.2 um | +9% | no (met on dev/select only) |
| Coverage >= 95% of baseline | 100% | 99.7% | yes |
| False two <= 5% with interval | 75.1% (P11) | 2.2% [1.7, 2.9] | yes |
| Resolvable recall >= 90% | 77.0% | 49.7% | no |
| Simultaneous recall >= 85% | 64.0% | 35.1% | no |
| Resolvable triples >= 40% | 0% | 0% | no |
| ~90% complete-population coverage | 81.6% (P11 protocol) | 85.7% (53% unbounded); 90.5% at 80% acceptance | no (bounded) |
| Single-source latency <= 1 ms | 0.6-3 ms | 0.90 ms | yes |
| Multi-source latency <= 2 ms | 8.6 ms | 5.65 ms | no |
| Sustained throughput above the event rate | - | 1 core 0.77x, 2 cores 1.5x | with 2 cores |

## Recommendation

Use unit-level pooled positions, with drift tracking and reported early/late disagreement, as NeuroSig's spatial
output. Use single-event positions with identifiability-scaled uncertainty. Treat source counting as unresolved: K = 1
plus an overlap probability by default. Validated progress on overlaps needs an extended-source (multi-compartment or
learned) single-neuron model, and ground truth from paired juxtacellular or hybrid recordings, rather than more
line-source variants.
