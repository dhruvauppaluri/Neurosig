# Phase 11 validation: real-neuron localization reliability

Date: 2026-11-07. Branch: `phase11-localization`, cut from `9c29118`. Software only. No RTL, FPGA, fixed-point
detector, frozen Phase 6 detector or `edge/localize/*` file changed, and no new beamforming family was added. Every
number below is read from the JSON files in this directory. The log is `experiment_registry.jsonl` (19 entries); the
method write-up is `docs/phase11-localization.md`.

**Measures kept separate:** synthetic true-coordinate error, sorter-associated disagreement, within-unit spatial
stability, detector-to-sorter agreement, multi-source recovery and calibration coverage. They are never combined into
one score.

## 0. Data and protocol

| Role | Data | Status |
|---|---|---|
| Development | Phase 5 calibration + selection windows, sessions A–C (24 windows) | used throughout |
| Model selection | Phase 5 held-out (A–D) + Phase 6 E–H + Phase 10 I–L (68 windows) | **previously used**; historical sets, not fresh holdouts |
| Final | **M** mainenlab 0b8ea3ec, **N** hoferlab 19baa84c, **O** zadorlab 1bd85dab, **P** steinmetzlab 0aafb6f1 | 24 new windows, reserved before any Phase 11 measurement (seed 20271107), evaluated once (`final11.json`). One candidate was skipped because of a degenerate sync file. |
| Synthetic | New pools p11dev (calibration / decisions), p11select (comparison), p11final (single use) | |
| Biophysical | 72 MEArec/NEURON templates (6 BBP L5 cell types) on Neuropixels-128 with known soma positions | independent of every NeuroSig forward model |

The frozen configuration is **P11** (hash `188c8efd636e`, `edge/phase11/configs.py`). Its components and calibration
table were fixed before the final windows were evaluated.

## 1. Phase 11A: event referencing (the checkpoint)

The three referencing strategies were localized with the same engines (P10, identical to P9 on intact contacts) and
compared on identical events. The matching rule was fixed in advance: optimal one-to-one assignment, ±0.5 ms,
detector channel within 60 µm of the sorter channel. Chance is estimated with shifted references.

| Split | Sorter events | Detector events | Matched | Chance | Detector − sorter time |
|---|---|---|---|---|---|
| Dev | 12,872 | 12,625 | 5,900 (46%) | 10% | median 0 ms, IQR −0.03 to +0.07 ms |
| Selection | 41,841 | 34,983 | 19,105 (46%) | 10% | |
| Final M–P | 23,469 | 11,761 | 7,508 (32%) | 8% | |

**Matched-event within-unit spread** (weighted scan; median spread / p90 deviation in µm; the +20 ms null is about
75–80 µm in every case):

| Split | A: sorter | B: detector | C: aligned ±0.2 ms | B vs A, median (95% CI) |
|---|---|---|---|---|
| Dev (125 units) | 14.4 / 74.4 | 14.4 / 50.8 | 15.7 / 51.7 | +0.5% (−19, +23) |
| Dev, E3 rule (align on the engines' own signal, ±3 samples) | 14.4 / 74.4 | | 14.9 / 58.8 | |
| Selection (445 units) | 11.9 / 47.4 | 13.8 / 44.7 | 13.9 / 43.2 | **+16% (+7, +26)** |
| **Final (214 units)** | 12.1 / 55.0 | 12.7 / 49.9 | 12.8 / 48.3 | +5% (−6, +16) |

**Full stream on final M–P:**

| Measure | Sorter events (A) | Detector events (B) |
|---|---|---|
| Events with a measurable trough (SNR ≥ 4) | 39.2% | 88.1% |
| +20 ms null, same measure | 8% | 9% |
| Median SNR | 3.4 | 5.1 |
| Spread, all labelled sorter events | 17.7 µm | |

Coverage (fraction localized) is ≥ 0.99 for every strategy.

**Timing and SNR perturbations** (dev, clean events with SNR ≥ 8; `decompose_11a_dev.json`):

*Timing.* Shifting the event centre moves the estimate:

| Shift | Median movement | p90 |
|---|---|---|
| 1 sample (0.033 ms) | 5.7 µm | 27 µm |
| 2 samples | 12.4 µm | 74 µm |
| 3 samples | 24 µm | |
| 6 samples | 59 µm | |

*SNR.* The same clean spikes were degraded to a lower SNR by mixing in real quiet background. The resulting
displacement is set against the deviation actually observed at that SNR:

| SNR | Noise-only displacement | Observed deviation |
|---|---|---|
| 2–3 | 27 µm | 49 µm |
| 3–4 | 16 µm | 28 µm |
| 4–6 | 8.6 µm | 18 µm |
| 6–8 | 5.2 µm | 10 µm |

Noise alone explains about half of the observed deviation at each SNR.

*Reference timing.* Events that the alignment rule would move by 3–5 samples deviate 39 µm, and by 6 samples
56 µm, against 14 µm for events it leaves in place. 46% of sorter events have no measurable trough.

**Answer at the checkpoint.** NeuroSig is limited first by **signal quality and the reliability of the reference
events**, second by sample-level timing jitter, and least by the spatial model.
- Detector referencing and local alignment both cut the p90 tail: 9–32% on dev and final, though only 6–9% on
  selection.
- They do not lower the median. On selection the median got significantly worse, because sorter template-fit times
  are a steadier within-unit timebase than single-trough detector times.
- Detector referencing therefore did **not** give the substantial stability gain the checkpoint required, and per
  your rule the forward-model and overlap work proceeded, kept bounded.
- Targets: matched-spread −20% **missed** (final +5%). p90 −15%: **missed** on final for B (−9%) and C (−12%), met
  on dev only. Null control: the improvement does not appear in the null. Coverage: met.

## 2. Phase 11B: footprints and forward models

**Footprint study** (dev, 43 units, `exp11b_dev.json`):
- Median 26 contacts significant per unit, 14 of them dominated by the positive phase.
- Axial asymmetry log-ratio +0.29 (larger above the peak than below).
- Spatial extent 77 µm at the trough, 84 µm at repolarization.
- Trough width 0.17 ms.
- The λ=20 line source leaves **54%** of the unit-template energy unexplained.

**Model comparison:**

| Model | Held-out-contact error vs line20, dev LOSO | Within-unit spread vs line20, dev | Same, selection (364 units) | Biophysical soma error, SNR 8 (median / p90) |
|---|---|---|---|---|
| line20 (baseline) | 0 | 0 | 0 | 15.1 / 35.7 µm |
| linefree (free distance) | −4% | −14% (CI −30, +12) | −5% (CI −12, +1) | 14.5 / 34.8 |
| radial (empirical) | −11% | +34% | +27% | 13.4 / 35.1 |
| aniso (empirical 2-D) | −18% | +72% | | 14.9 / 34.6 |
| twoline (two components) | −24% | +50% | | 15.2 / 31.6 |
| line20c (line + offset) | −37% | +23% | +28% | 14.0 / 36.6 |

On final M–P, linefree changed spread by −5% (147 units).

**Gate 11B: no new model retained.** Every flexible model reconstructs held-out contacts better and localizes the
same unit **less** consistently. The extra freedom absorbs waveform variation into position. The free-distance line is
the only consistent small gain (−5% on selection and final). That is below the −15% target and costs 2× the runtime.
Targets: −20% reconstruction met by twoline and line20c, but only with a stability loss. −15% spread missed.

## 3. Phase 11C: difficult sessions (`exp11c_sessions.json`, `exp11c_good_vs_other.json`)

Across the 12 historical sessions, the full-stream spread is correlated (Spearman, n=12, descriptive) with:

| Factor | ρ |
|---|---|
| Fraction of IBL-"good" units | −0.82 |
| Median SNR | −0.80 |
| IBL amplitude | −0.74 |
| Trough fraction | −0.72 |
| Spike width | +0.65 |
| Contact noise | −0.34 |
| Overlap rate | −0.38 |

Sessions F, G and K combine low amplitude (about 30 µV), median SNR 2.8–3.5, troughs on only 14–36% of sorter events
and 4–8% "good" units. F and K also have the widest waveforms (0.57 and 0.30 ms, against 0.17–0.23 ms elsewhere).

**Decisive check: IBL-"good" units localize stably in every session.**

| Session | Good units: spread (n) | Other units: spread |
|---|---|---|
| F | 4.0 µm (n=2) | 54 µm |
| K | 4.8 µm (n=1) | 56 µm |
| All 12 sessions | 4–11 µm | |

The hard sessions are hard because their reference units are weak and poorly isolated. The localizer is not the cause.

**Taxonomy and interventions:**

| Category | Sessions | Intervention tested | Result |
|---|---|---|---|
| Unreliable reference / poor isolation | F, G, K (final: M, N) | Detector referencing | Matched spread falls on F (26.7 → 22.4 µm), K (58.3 → 41.3) and final N (34.7 → 19.8). G (+7%) and N's p90 get worse. Mixed. |
| | | Weak-trough flag (reported, not removed) | Adopted in P11 |
| Low SNR | F, G, K | Free-distance model | F −11%, K −7%. Small. |
| Timing jitter | all | Local alignment | Tail −9 to −12% on final; median unchanged |
| Dead / noisy contacts | A (14 flat contacts) | 11D fallback | See §4 |
| Dense overlap | E, L | None needed | These sessions are among the stable ones |

The fixed "poor isolation < 25% good" threshold flagged 11 of 12 sessions, so it is not discriminating. This is
reported as is.

## 4. Phase 11D: dead-contact fallback (`exp11d_dropout.json`, `final11.json`)

The rule was chosen on dev by a pre-registered criterion: reject when the interpolate (P9) and exclude (P10)
estimates disagree by more than 80 µm. Unknown dropout covers random 25%, random 50% and clustered 25% (P9/P10
engines, three engines).

| Split | P9: median / p90 | P10: median / p90 | **P11: median / p90 / accepted** |
|---|---|---|---|
| Dev | 13.0 / 89.6 | 7.3 / 81.2 | 7.0 / 73.3 / 96.1% |
| Selection | 13.1 / 100.9 | 7.8 / 109.5 | 7.0 / 75.1 / 93.5% |
| **Final** | 14.0 / 111.1 | 7.8 / 108.4 | **7.3 / 91.7 / 94.1%** |

By condition on final, the P11 p90 changes against P10 by −26% (random 25%), −12% (random 50%) and −10%
(clustered).

Targets on final:
- Keep ≥80% of the median gain: **met** (108%).
- p90 −20% against P10: **missed** (−15.4%; met only in the random-25% condition).
- Coverage within 5 points: **missed by 0.9** (94.1%; rejection applies only to events with dead contacts).

On selection, "interpolate when fewer than 2 live contacts are near the estimate" would have done better (p90 −17%
with no rejections). It was not chosen, because dev data showed no benefit.

## 5. Phase 11E: source count (`exp11e_overlap.json`, `final11.json`)

**Change tested.** OV-A applies the boundary veto only at the axial ends. In the frozen pipeline, a source fitted at
the lateral limit cancelled genuine pairs; this was 8 of 15 failures on dev. OV-B, a misfit-normalized count statistic
(single-source χ²_red is about 4.4, not 1), gave no gain on selection and was rejected. K=3 stays experimental.

| Pool | Rule | Resolvable recall | All-pair recall | False two-source (singles) | Simultaneous resolvable | Triples, all three | Per-source error |
|---|---|---|---|---|---|---|---|
| Selection | P10 | 92.2% | 54.6% | 3.3% | 69.6% (n=23) | 0% | 3.6 µm |
| Selection | **OV-A** | 94.1% | 55.6% | 4.2% | 73.9% | 0% | 3.7 µm |
| Selection | OV-A+K3 (experimental) | 94.8% | 54.4% | 4.2% | 78.3% | 20.0% (2% false triples on pairs) | 3.7 µm |
| **Final** | P10 | 92.6% | 53.5% | 8.3% | 66.7% (n=24) | 0% | 3.7 µm |
| **Final** | **P11 (OV-A)** | 92.6% | 54.4% | **9.2%** | 66.7% | 0% | 3.7 µm |

Targets on final:

| Target | Result |
|---|---|
| Resolvable recall ≥ 90% | **Met** (92.6%) |
| False two-source rate ≤ 5% | **Missed** (9.2%; 4.2% on selection) |
| Per-source error ≤ 10 µm | **Met** (3.7 µm) |
| Simultaneous-pair recall ≥ 85% | **Missed** (66.7%) |
| Triples ≥ 40% all recovered | **Missed** |

Across five fresh 120-single pools (Phases 10 and 11), the false two-source rate ranges from 3.3% to 9.2%, wider than
any rule change. At this sample size the rate is not well determined, and OV-A neither helps nor clearly hurts it.

## 6. Phase 11F: conditional calibration (`exp11f_calibration.json`, `final11.json`)

Mondrian split-conformal over source-count stratum × SNR group (<8, 8–16, ≥16) × dropout, fitted on p11dev. Groups
with fewer than 30 calibration events use the larger of their own and their stratum quantile.

| Pool | Method | Coverage (95% CI) | Low / mid / high SNR | Dead contacts / none | Median area | Confident-but-large-error rate |
|---|---|---|---|---|---|---|
| Selection | Phase 9 frozen | 87.9% (85.8–89.8) | 78.2 / 92.9 / 94.7 | 85.4 / 90.5 | 6,668 µm² | 0.68% |
| Selection | Mondrian | 90.4% (88.4–92.0) | 90.7 / 89.7 / 90.9 | 91.4 / 89.3 | 1,535 µm² | 0.10% |
| **Final** | **Mondrian** | **90.5% (88.5–92.1)** | 91.8 / 90.6 / 88.2 (83.5–91.7) | 90.7 / 90.2 | 1,629 µm² | 0.0% |

On final, coverage by stratum was clean 90.8%, flagged 83.0%, two-source 92.4% and shared 91.1%. The accepted fraction
was 91.7%.

**Risk–coverage on final**, ranking events by region size and keeping the smallest:

| Kept | Median error | Rate of errors > 50 µm |
|---|---|---|
| 50% | 3.4 µm | 0.8% |
| 70% | 4.5 µm | 1.7% |
| 90% | 5.7 µm | 4.8% |
| 100% | 6.5 µm | 6.6% |

**Caveat.** Coverage is scored on accepted events whose primary estimate lies nearest the event's reference source,
which is 77% of accepted events on final. Primaries nearest the other source of a pair are not scored.

Targets: overall ≈90% **met**. Each adequately sampled SNR group ≥90%: **met in point estimate for low and mid**; high
is 88.2% with a CI that includes 90%. Area: **reduced 76%**. Accepted fraction ≥90%: **met**. Confident-but-large
errors reduced ≥25%: **met** (0.68% → 0.10% on selection, 0% on final).

## 7. Phase 11G: runtime (`exp11g_runtime_*.json`, Phase 10 `runtime_memory.json`)

- **Single-source P11 latency:** 0.23–1.0 ms median per engine (0.6 ms for weighted scan), about 1,600 events/s on one
  core.
- **Real event rate:** the measured dev detector rate is 263 events/s per 32-contact section, a 6× margin for
  single-source localization.
- **K≤2 overlap pipeline:** 8.7 ms median, 19 ms p99, about 106 events/s. That is **not** sustainable for every event
  at 263 events/s on one core.
- **Staged gating failed.** Two cheap spatial ambiguity gates were tried: maximum standardized residual (G1), and a
  greedy second-source F-ratio (G2). Holding ≥98% of the full pipeline's two-source declarations required invoking the
  fit on about 94% of events, so the staged pipeline cost slightly more than always fitting. Pairs offset in time are
  separable mainly through temporal waveform differences, which a single-sample spatial score cannot see.
- **Accuracy–latency trade-off:** single-source only takes 0.6 ms with no overlap handling; the K≤2 pipeline takes
  about 9 ms. ≤2 ms with refinement is **not achieved**.
- **Memory:** P11 bounds the dead-contact cache to 4 masks. Resident memory after 40 distinct masks is 41 MiB, against
  165 MiB with Phase 10's 16-mask cache; outputs are identical.

## 8. Final engineering decision

1. **Dominant limitation.** Signal quality and reference reliability.
   - 61% of final sorter events have no measurable trough at the sorter time.
   - Deviation rises from 7.6 µm in the highest SNR quintile to 68 µm in the lowest.
   - Well-isolated units localize at 4–11 µm in every session.
   - Sample-level timing jitter is second; the spatial model and overlap are minor contributors on real data.
2. **Stability on independent recordings.** Not improved in the median.
   - Final matched-event spread is 12.1 µm (sorter) against 12.7 µm (detector).
   - The p90 tail falls 9–12% with detector referencing or local alignment.
   - Dead-contact robustness improved: the median shift is about half P9's (14.0 → 7.3 µm), and the p90 is 15% lower
     than P10's at 94% coverage.
   - Worse cases: on N, detector referencing worsens the p90 (115 → 197 µm). On selection, the detector-referenced
     median is +16%.
3. **One source versus several.** Resolvable pairs are recovered at 92.6% with 3.7 µm per-source error. However, the
   false two-source rate on single events is 3–9% depending on the pool (final 9.2%). Simultaneous pairs reach 67% and
   triples are not resolved. Not yet reliable enough to claim.
4. **Uncertainty estimates.** Trustworthy in the conditions tested.
   - Coverage on final is 90.5% (CI 88.5–92.1), roughly 90% in each SNR group, with dropout and without.
   - Regions are 76% smaller than Phase 9's, and confident-but-large errors fall to zero.
   - Limits: synthetic events from one forward family, and coverage is scored on the 77% of accepted primaries nearest
     their reference source.
5. **Edge practicality.** Single-source P11 is practical: under 1 ms per event, a 6× margin at the measured rate and
   41 MiB bounded memory. The K≤2 fit at about 9 ms is not sustainable for every event, and cheap gating could not
   reduce it.
6. **Unresolved.**
   - There is no independent position ground truth on real data.
   - The sorter reference is unreliable for poorly isolated units.
   - The false two-source rate varies across pools.
   - Simultaneous and three-source overlaps are not resolved.
   - The K≤2 latency needs temporal-domain gating or compiled kernels.

**Recommended next experiments:**
- Real-data evaluation restricted to, or stratified by, sorter quality labels.
- Detector-referenced pooling of events per putative unit.
- A larger single-source pool to pin down the false two-source rate.
- A temporal-domain ambiguity gate.

**What failed:** detector and aligned referencing for the median; every flexible forward model (radial, aniso,
twoline, line20c); the misfit-normalized count statistic; K=3 at ≥40%; both runtime gates.
