# Phase 10 validation: localization gap-closing and fresh locked evaluation

Date: 2026-11-07. Software only. No RTL, FPGA, fixed-point detector, or `edge/localize/*` file changed, and no new
beamforming family was added. Methods are in `docs/phase10-localization.md`. Every number below is read from the JSON
files in this directory. The experiment log is `experiment_registry.jsonl`.

## 0. Why Phase 10 exists

The localization brief re-sent on 2026-11-07 mostly repeats work already done in Phase 8 (`838a034`) and Phase 9
(`b6665fe`). The user chose to close the gaps rather than re-run everything. Phase 10 therefore covers only what is
missing:
- stability across recording segments,
- how error depends on amplitude and SNR,
- removal of the highest-amplitude electrode,
- three-source events, partial temporal overlap, and dropout during overlap,
- a memory benchmark,
- a fresh locked evaluation and decoding on new sessions.

Phase 9 had already spent locked sessions E–H, so Phase 10 reserves four new insertions:

| Key | Lab | PID (prefix) | Notes |
|---|---|---|---|
| I | wittenlab | 0851db85 | |
| J | cortexlab | 0a0184b3 | |
| K | mrsicflogellab | 143dd7cf | |
| L | churchlandlab_ucla | 02cc03e4 | Only lab already used before (in session H); new animal and insertion |

The set is 24 two-second windows, drawn with seed 20271106. They were reserved before any window was downloaded. Two
candidates, churchlandlab 0393f34c and mainenlab 071f02e7, were skipped for availability: their probe sync file has
only two rows, so the probe-clock correction cannot be validated.

All 24 windows downloaded; there were no `missing_dataset` entries. The set contains 1,437 sorter events (I 359, J 358,
K 360, L 360) and 101 units with at least 5 events.

## 1. Configurations

| Name | Hash | What |
|---|---|---|
| legacy raw | – | Existing five engines on raw snippets (historical path) |
| corrected B | – | Existing engines on baseline-subtracted snippets. This is the null-controlled baseline from Phase 9. |
| P9 | `5ed8a69b97eb` | Frozen in Phase 9: baseline subtraction, CAR, dead-contact interpolation, λ=20 µm line source, noise-aware robust rule |
| **P10** | `8a56becbfc85` | P9 with dead contacts excluded (engine rebuilt on the live contacts) instead of interpolated. Identical to P9 when no contact is dead. Frozen from dev windows A–D and synthetic pools before I–L was downloaded. |

## 2. Development findings (windows A–D, `real_extra_dev.json`)

- **Segments.** Within-unit spread is similar in the early, middle and late thirds of each session (P9 weighted scan:
  17.0, 18.0 and 19.7 µm). The largest shift between segment medians is 23.7 µm, against a null floor of 66.5 µm. No
  recording-drift effect is visible above event-level scatter.
- **SNR dominates instability.** Deviation from the unit median falls about 4× from the lowest to the highest SNR
  quintile: P9 weighted 42.4, 32.7, 21.1, 18.0 and 9.6 µm (Spearman ρ −0.32). The lowest quintile has a sorter-channel
  trough SNR of about 0.2, meaning no visible spike at the sorter time. Low-SNR sorter events explain the remaining
  within-unit spread better than estimator choice does. Rejecting them would lower the median only by dropping events,
  which the brief forbids, so nothing was changed on that basis.
- **Peak-electrode removal is benign** for P9: median shift 3–4.5 µm whether the contact is known or unknown to be
  missing. Legacy `robust_scan` with the peak contact unknown-missing moves a median 84 µm (dev) and 91 µm (corrected B,
  locked).

## 3. Experiments (budget: 3 algorithm changes; the 30-experiment budget across Phases 8–10 is not exhausted)

| ID | Hypothesis | Result | Decision |
|---|---|---|---|
| E1 | A greedy third source orthogonal to the accepted K=2 span recovers triples | Triples: all three recovered 0% → 20% (resolvable triples 0% → 46%). False triples on resolvable pairs 1.3%. Singles unchanged. | Retained as an opt-in extension; off in the frozen pipeline |
| E2 | Interpolated dead contacts look like a second source to the K≤2 fit; masking them removes phantoms | Confirmed on dev and on a fresh confirmation pool (below) | Retained for the overlap fitter |
| E3 | Rebuilding the P9 engines on the live contacts beats interpolation under unknown dropout | Dev unknown-dropout median shift: −16% to −47% across all five engines; p90 −3% to −34% | Retained, frozen as P10 |

E2, with masks identical across arms (`exp_dead_mask_p10confirm.json`, a single run on a pool never used before):

| Unknown dropout | Pair recall, interpolate → mask | Single-source false two-source rate, interpolate → mask |
|---|---|---|
| none | 93.2% → 93.2% | 7.5% → 7.5% |
| random 25% | 77.7% → 89.9% | 59.2% → 5.8% |
| random 50% | 54.7% → 77.0% | 83.3% → 9.2% |
| clustered 25% | 79.1% → 81.1% | 80.0% → 10.0% |

## 4. Locked evaluation on I–L (`final_locked10.json`, evaluated once)

Within-unit spread is the median over 101 units, reported with the real/null ratio. Axial disagreement against the
sorter peak contact is a proxy, not an error.

| Method | Spread B → P9 (µm) | Change | Real/null P9 | Axial median B → P9 (µm) | Valid B → P9 |
|---|---|---|---|---|---|
| amplitude | 25.9 → 24.9 | −4% | 0.34 | 32.4 → 26.1 | 1.00 → 1.00 |
| template | 23.7 → 22.6 | −5% | 0.29 | 40.0 → 31.3 | 1.00 → 1.00 |
| beamform | 22.5 → 20.6 | −8% | 0.26 | 32.5 → 22.4 | 1.00 → 1.00 |
| weighted_scan | 20.9 → 20.0 | −4% | 0.26 | 25.0 → 20.0 | 1.00 → 1.00 |
| robust_scan | 29.5 → 24.1 | −18% | 0.29 | 45.1 → 22.9 | **0.78 → 0.99** |

- **Spread target missed.** Best-vs-best (weighted scan to weighted scan) is −4.3%, against the −25% target. Phase 9
  measured −9.5% on E–H. Coverage did not fall; `robust_scan` gains 21 points of coverage.
- **P10 equals P9 on intact data.** Its spread, axial and noise columns are identical to P9.
- **Legacy raw** weighted scan again shows a spread (6.2 µm) almost equal to its null (6.8 µm). This repeats the
  Phase 9 finding that raw-path stability is an artifact of static per-channel offsets.

Per session (weighted scan spread, B → P9):

| Session | B → P9 (µm) | Real/null (P9) | Note |
|---|---|---|---|
| I | 15.2 → 12.4 | | |
| J | 7.6 → 8.8 | | P9 worse |
| K | 57.1 → 50.1 | 0.62 | Poor for every method, like session F in Phase 9 |
| L | 11.7 → 13.7 | | P9 worse |

The improvement is not consistent across sessions.

**Dropout (unknown dead contacts, median / p90 shift in µm):**

| Engine | Random 25%: B / P9 / P10 | Random 50%: B / P9 / P10 |
|---|---|---|
| weighted | 9.3/79, 10.0/78, **5.9/116** | 24.1/157, 22.4/152, **16.1/227** |
| beamform | 11.6/109, 11.0/78, **6.6/131** | 26.5/205, 23.9/157, **18.6/247** |
| robust | 101.8/497, 11.5/87, **6.9/124** | 108.3/533, 25.3/172, **19.9/242** |

- **P10 against B, median:**
  - Random 25%: −28% to −56% on every engine (robust_scan −93%).
  - Random 50%: −29% to −33% for template, beamform and weighted; −11% amplitude; −82% robust.
  - Clustered 25%: four engines improve; weighted gets worse (4.1 → 4.5 µm).
- **P10 against P9, median:** −6% to −55% depending on engine and level. Weighted scan improves −41% at 25% and −28%
  at 50%.
- The ≥20% target is met **in the median for most engine/condition pairs, not all**.
- **The p90 is worse on the locked set** (for example, weighted 78 → 116 µm), although it improved on dev. Rebuilding
  on fewer contacts sometimes produces a larger outlier than interpolation. This is a real regression in the tail and
  is reported, not tuned away.

**Noise (P9 vs B, p90 µm, weighted scan):**

| Noise type | B → P9 | Change |
|---|---|---|
| independent | 145 → 141 | −3% |
| correlated | 31 → 16 | −48% |
| impulsive | 50 → 51 | 0% |
| gain | 58 → 50 | −14% |
| drift | 206 → 199 | −3% |

The ≥20% target is **met only for correlated noise**.

**Segments, SNR and peak removal on the locked set (`extra_real`):**
- The segment shift median is 26.5 µm for P9 weighted (null 103.5).
- Error rises at low SNR on every engine (ρ −0.38 to −0.57). P9 weighted deviation by quintile is 77.5, 43.7, 24.5,
  9.9 and 6.6 µm.
- Removing the peak electrode shifts P9 by a median of 3–5 µm.

The development conclusions hold on new labs.

## 5. Overlap stress (`overlap_stress.json`, p10test pool, single run)

The frozen Phase 9 K≤2 pipeline was used with its frozen calibration.

**Fresh-pool baseline:**
- Resolvable pairs: 89.3% have both sources recovered and accepted, per-source error 3.5 µm.
- Singles: false two-source rate **7.5%**. The p10dev and p10confirm pools gave 8.3% and 7.5%.

The Phase 9 final figure (3.2%) was optimistic, and the ≤5% target is not robustly met.

**Temporal overlap, resolvable pairs (all sources recovered):**

| \|dt\| | Recovered |
|---|---|
| 0 samples | 76.9% |
| 1–3 samples | 87.5% |
| 4–9 samples | 95.8% |
| 10–20 samples | 100% |

Fully synchronous spikes are the hardest case.

**All pairs by noise.** At 20 µV noise, recovery is 0–20% at every offset, because most such pairs fall outside the
resolvable definition.

**Separation, all pairs:**

| Separation | Recovered |
|---|---|
| <40 µm | 34% |
| 40–60 µm | 57% |
| 60–90 µm | 63% |
| ≥90 µm | 52% |

**Triples:**
- K≤2 pipeline: never recovers all three. It finds 1.43 of 3 sources on average, and 18% carry a "possible
  unresolved overlap" flag.
- E1 extension: all three recovered on 20% (46% of resolvable triples), with mean matched error 7.8 µm.

**Dropout during overlap, Phase 9 interpolation behaviour:**

| Dropout | Pair recall | Single false two-source rate |
|---|---|---|
| Known random 25% | 81.3% | |
| Known random 50% | 69.3% | |
| Unknown random 25% | 72.7% | 52% |
| Unknown random 50% | 49.3% | 83% |
| Unknown clustered 25% | 78.0% | 87% |

E2's mask fixes the unknown cases (section 3).

## 6. Decoding on the new insertions (`decode_new_sessions.json`, frozen Phase 9 protocol)

Change from adding P9 locations to counts, in points, with 95% CI:

| Session | Region | Trials | Change (pts) | 95% CI |
|---|---|---|---|---|
| I | CA1 | 559 | −0.4 | [−1.2, +0.2] |
| J | SUB | 396 | −0.1 | [−1.3, +1.0] |
| K | LP | 612 | −0.3 | [−1.0, +0.2] |
| L | APN | 511 | +0.2 | [−0.6, +1.0] |

- Counts alone reach 51.7%, 52.8%, 69.9% and 73.5% (majority 53.0%, 54.3%, 59.3% and 51.5%).
- Position features are again explained by the counts (median out-of-fold R² ≈ 1.0).
- MLP with real versus shuffled positions at identical capacity: +1.3, −0.1, −0.7 and +0.5 points; every CI includes 0.
- The +3 point target is **missed on all four new sessions**. Together with Phase 9's eight sessions, 12 insertions
  show no evidence that unit-level localization adds information to left/right stimulus decoding beyond spike counts.
- The most-populated region on these probes is often not visual cortex, so the task offers little spatial signal to
  find.

## 7. Runtime and memory (`runtime_memory.json`, Apple Silicon, single thread)

Median latency per event, with p99:

| Method | P9 | P10 |
|---|---|---|
| amplitude | 0.91 ms (p99 12.6) | 0.82 ms (p99 1.3) |
| template | 0.18 ms | 0.23 ms |
| beamform | 0.32 ms | 0.34 ms |
| weighted | 0.54 ms | 0.60 ms |
| robust | 1.10 ms | 1.01 ms |
| K≤2 overlap pipeline | 9.0 ms (p99 23.0) | |

- **Legacy raw comparators:** 0.65, 0.03, 0.14, 0.44 and 0.78 ms for the five methods in the same order.
- **P10 with eight dead contacts:** the first call costs 5–7 ms (engine rebuild), then 0.2–1.0 ms.
- **Peak transient memory per event:** about 5.0 MB for weighted and robust scans, 1.7 MB amplitude, 0.18 MB beamform,
  0.06 MB template, 0.77 MB overlap pipeline.
- **Resident memory:** the five-engine factory holds about 12.6 MB. Each cached P10 dead-contact mask adds about 9.8 MB
  across the five engines. The worst case of 16 cached masks is about 157 MB, acceptable on a host but not on an
  embedded target.
- **Ratios:** P9 template takes 3.6× its corrected-B time (0.05 → 0.18 ms), the only engine over 2×.
- The earlier timing pass overlapped with another job and was discarded; these numbers come from an isolated rerun.

## 8. Targets

| Target | Status |
|---|---|
| Within-unit spread −25% with ≤5-point coverage loss | **Missed** (−4.3% best-vs-best on I–L; coverage +0 to +21 points) |
| Dropout instability −20% | **Mostly met in median** (P10 vs B −28% to −56% at random 25%; not every engine at 50% or clustered); **p90 worse** on locked |
| Noise p90 −20% | **Missed** except correlated noise (−48%) |
| Improvement across multiple recordings | **Not consistent** (2 of 4 new sessions worse for weighted scan) |
| Two-source recall ≥90% on resolvable pairs | **Borderline** (89.3% on the fresh pool; 94.5% in Phase 9) |
| False two-source rate ≤5% | **Missed** on three fresh pools (7.5–8.3%) |
| Per-source median error ≤20 µm | **Met** (3.5 µm) |
| Overlap under dropout | **Fixed** for unknown dropout by E2 (false two-source rate 52–87% → 6–10%) |
| Triples | Partial (E1: 20% all-recovered, 46% of resolvable) |
| Decoding +3 points | **Missed** (12 insertions, null) |
| Runtime ≤2× | Met except template (3.6×, absolute 0.18 ms) |

## 9. Limitations and next steps

- Sorter-referenced positions are proxies.
- The remaining instability is mostly sorter events with no measurable trough at the sorter time (SNR < 1). Better
  estimators cannot fix that without discarding events. A detector-referenced event set, or per-unit pooling of
  events, would answer it.
- Real-like synthetic data uses the footprint that P9 was calibrated on, which gives P9 a model advantage.
- Session K (like F before it) is poor for every method; its cause is not established.
- P10's tail regression under unknown dropout needs a fallback rule, for example interpolate when more than X% of
  contacts are dead. That would be a new experiment on development data, not on I–L.
- The I–L locked set is now spent for P9 and P10.
