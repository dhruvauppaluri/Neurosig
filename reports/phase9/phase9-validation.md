# Phase 9 validation: corrected real-data localization, stability, overlap, uncertainty, decoding

(The task brief called this "Phase 8"; that number was already used by `reports/phase8/`, so this work is **Phase 9**.)
Software only. No RTL, FPGA, detector, fixed-point or `edge/localize/*` code changed; every new engine wraps the existing ones.
Branch `phase9-realdata-localization`. Frozen configuration hash `5ed8a69b97eb` (`edge/phase9/configs.py`).
Experiment log: `reports/phase9/experiment_registry.jsonl` (17 experiments, below the 30 budget; stopped under "repeated changes no longer improve development objectives" and "further gains need a different algorithm family").
Proxy caveat throughout: sorter peak-channel depth and unit-level consistency are proxies, **not** neuron coordinates.

## 1. Corrected baseline (Phase 1) — `reports/phase9/corrected_baseline.json`
* **Timestamps traced.** Raw samples are probe-clock; cached windows store sorter times mapped through `_spikeglx_*.timestamps.npy` (`sync_version` 3). `edge/phase9/events.py` refuses un-synced windows and exposes the historical session-clock mistake only as an explicit control.
* **Reproduced.** Pilot (4836a465, t0=100 s, first 50 sorter spikes), median axial disagreement (µm), corrected vs published Phase 6: amplitude 21.68 vs 21.7, template 20.0 vs 20.0, beamform 19.40 vs 19.4, robust_scan 22.60 vs 22.6 (all within 0.05; tolerance 0.5). The session-clock control reproduces the originally published errors (40.6 / 90.0 / 68.5 / 48.5). `weighted_scan` (not previously published for the pilot): 17.5 corrected vs 40.0 control. Note `ibl.py` and `ibl_eval.py` configure `robust_scan` differently (offset_reciprocal vs line_source); both are labelled.
* **Waveform alignment.** On filtered traces (frozen CAR 300–6000 Hz) the trough on the sorter peak channel is at median offset 0.0 ms after correction (session-clock control: 0.2 ms median, **1.5 ms median |offset|**). Only 49.9% of corrected events fall within 0.2 ms of the trough and the 90th-percentile |offset| is 2.4 ms: sorter times are template-referenced, not trough-referenced, so a residual per-event timing scatter remains. It is not a clock error.
* **All five methods, identical events** (73 cached windows, 4,372 sorter-centred raw snippets, legacy raw path): median/p90/p95 axial disagreement (µm) amplitude 40.8/139.6/167.7, template 40.1/238.9/377.2, beamform 40.0/203.4/309.2, weighted 40.1/203.5/263.5, robust 41.5/186.7/216.4 with **66.6% valid**; median runtime 0.64 / 0.035 / 0.18 / 0.41 / 0.76 ms; method-to-method median disagreement 2.5 µm (beamform|weighted) to 29.8 µm (amplitude|robust). The pilot's ~20 µm agreement does not generalize: across windows and sessions the median is ~40 µm (one contact pitch), with heavy tails.
* **Tests** `edge/tests/test_phase9_alignment.py` (7): synthetic trough centring, 1.5 ms session-clock error visible, un-synced windows refused, drift round-trip, geometry, cached-window comparison.
* Historical Phase 3–4 / Phase 6 reports are unchanged; see `reports/DEPRECATIONS.md` for the superseded stability numbers.

## 2. A methodological finding that changes the baseline (null control)
Within-unit spread on the **raw** snippets looks excellent (amplitude 7.4 µm) but null events (same units' times shifted +20 ms, no spike) are just as "stable" (6.6 µm): the raw estimate is dominated by a static per-channel offset pattern. After pre-event baseline subtraction real events are 3–4× tighter than null (22–28 vs 67–92 µm) and keep their rank correlation with sorter depth (0.78 vs ~0). **Consequently the correct reference is "corrected B" (baseline-subtracted existing engines), the stability metric is real/null spread, and any earlier raw-path stability number is not evidence of localization quality.** (Phase 8's real-data table used a different filtered, trough-re-aligned path without a null control.)

## 3. Improvements (Phase 3) and what was tried
Retained and frozen as **P9**: (1) baseline subtraction, (2) snippet median-CAR, (3) dead-channel interpolation, (4) forward model calibrated to the measured real footprint (line source, λ = 20 µm; monopole λ = 30 µm for amplitude/template — the real footprint is ~4× more compact than the assumed 80 µm; `reports/phase9/real_footprint.json`), (5) noise-aware overlap rule for `robust_scan` (a second peak must clear 6σ).
Rejected with evidence: data-driven trough re-alignment (worse: 36.6 vs 19.0 µm spread), 3-tap temporal mean/median, rank-1 temporal-projection amplitudes (fails on real data again), per-channel noise normalisation (fixes gain, hurts everything else).

### Locked sessions E–H (24 windows, run once for P9), P9 vs corrected B on identical events
| Method | Axial median change | Within-unit spread change (same units) | p90 spread | Valid | Runtime |
|---|---:|---:|---:|---:|---:|
| amplitude | −18% | −17% | −16% | 100% | ×1.2 |
| template | −18% | −10% | −23% | 100% | ×2.9 (0.03→0.16 ms) |
| beamform | −24% | −16% | −8% | 100% | ×1.6 |
| weighted_scan | −21% | −7% | −13% | 100% | ×1.2 |
| robust_scan | −30% | −19% | −20% | 68%→**99%** | ×1.2 |
Best-vs-best median within-unit spread: 26.7 µm (weighted, B) → 24.1 µm (beamform, P9) = **−9.5%** (target −25%: **missed**). On the previously used Phase 5 held-out windows (labelled as such): −17.4%.
**Per-session (spread µm, B → P9), locked, beamform:** E 28→24, F 54→43, G 27→26, H 23→17; weighted: E 25→24, F 65→48, **G 26→28 (worse)**, H 16→17 (worse); robust: E 33→25, F 74→49, G 32→27, H 28→19. Session F (angelakilab) is the least stable for every method and configuration (43–51 µm) and was not investigated further. On the used held-out set weighted D 15→16 and robust C 24→30 are worse.
Absolute positions on locked sessions (P9 axial median 27.5–33 µm with p90 141–180 µm, vs sorter peak-channel depth) are only an agreement proxy.

## 4. Robustness (Phase 2 tests B and C), locked, P9 vs B (mean change in median displacement across conditions)
| Method | Known-dropout (random/clustered 75/50%) | Unknown-dropout (zeroed contacts) | Noise (5 types) | Noise p90 |
|---|---:|---:|---:|---:|
| amplitude | −20% | −37% | −23% | −2% |
| template | −17% | −41% | −10% | −19% |
| beamform | −27% | −33% | −11% | −15% |
| weighted | −22% | −29% | −10% | −17% |
| robust_scan | −41% | −82% | −38% | **+46% (worse tails)** |
Dropout target (≥20% less instability): met for amplitude, beamform, weighted, robust (known) and all methods for unknown dropout; template −17% misses the known case. Noise target (≥20% without more outliers): met only by amplitude; robust_scan improves medians but its p90 gets worse. Impulsive/gain/drift conditions are in `reports/phase9/final_locked.json`.

## 5. Synthetic accuracy with known coordinates (Phase 3 targets)
Real-like generator (`edge/phase9/synth_real.py`: measured footprint λ=20 µm, 1 ms spikes, DC offsets, shared noise, 4-column staggered real geometry; **model advantage caveat: P9's gain matches the generator's footprint by construction**), single-use final pool n=450, 50% contaminated, xy error (µm) median/p90/p95; large-error (>50 µm) rate:
| | amplitude | template | beamform | weighted | robust (accepted) |
|---|---|---|---|---|---|
| legacy raw | 13.6/35.7/50.9, 5.3% | 16.1/33.6/44.8, 4.3% | 12.9/31.8/41.0, 3.0% | 12.6/30.9/40.9, 3.7% | 21.5/40.8/49.9, 5.3% (76% accepted) |
| corrected B | 9.9/29.6/39.0, 2.0% | 13.7/29.5/39.9, 2.0% | 9.4/25.5/38.1, 1.7% | 8.3/25.3/34.5, 1.7% | 17.0/35.0/44.0, 4.1% (82%) |
| **P9** | 10.9/25.9/34.9, 1.0% | 12.5/28.6/34.7, 0.7% | **8.2/22.3/29.7**, 1.7% | **7.8/24.2/34.5**, 2.0% | 8.0/24.1/36.5, 2.1% (**97%**) |
Median ≤ 12 µm and p90 ≤ 30 µm are met by beamform, weighted and robust (and amplitude); template misses the median (12.5). Large-error frequency vs legacy raw: −81% amplitude, −84% template, −60% robust_scan, −43% beamform, −46% weighted.
**Not a general win:** on the earlier broad-footprint generator (λ=80 µm, 2.5 ms spikes, no offsets; Phase 8 final pool) P9 is far *worse* (e.g. beamform 40/69 µm, 30% large errors vs 17.8/44 µm, 6.7%), because snippet CAR and λ=20 are wrong there. The two synthetic generators disagree about the right model; the measured real footprint supports the real-like one, and the Phase 8 synthetic gains therefore should not be read as evidence about real recordings.

## 6. Overlapping sources (Phase 4) — frozen overlap config, final real-like pool (n=450: 217 singles, 233 pairs, 73 resolvable)
The Phase 8 free-waveform fitter did **not** transfer (E12: resolvable recall 18.5%). Diagnosis: shared noise and CAR residue look like a second source (single-source statistic median 253, E13 refuted the z-offset explanation). Fix: treat common mode as a projected nuisance instead of applying CAR (E14; single-source median 63). Final results:
* Resolvable-pair recall **94.5%** (target ≥90%: met; 93.5–95.2% at 5–10 µV noise; 20 µV has no resolvable pairs by definition); all pairs 60.9%.
* False two-source rate on singles **3.2%** (target ≤5%: met; 5.5% at 20 µV).
* Per-source median error on correctly resolved pairs **4.0 µm** (target ≤20: met). Rejected 4.4% of events (2.7% of resolvable pairs), all with reasons.
* Large-error reduction ≥25%: **not met** — the K≤2 primary-source large-error rate is 12.3% vs 2.7% for the single-source P9 engine, and pair-event p90 is 62 µm vs 22 µm (the single engine is not penalised for the missed second source, so the comparison is unfavourable to K≤2 but still shows its tails come from mis-split non-resolvable pairs).
Triple-source and electrode-dropout-during-overlap cases were **not** evaluated in Phase 9 (Phase 8's triple results, 1.7 of 3 sources found on the legacy generator, stand and have not been re-validated).

## 7. Uncertainty (Phase 5), same final pool
Conformal 90% regions on the primary source, accepted events: coverage **88.6%** overall (singles 91.9%, pairs 85.5%; by noise 92% / 93% / 80% at 5/10/20 µV) — approximately but not conditionally calibrated, short of 90% at high noise. Accepted fraction 95.6% (≥90% met). Median region area 2,890 µm². Risk–coverage by region size (keep 50/70/90/100%): median 3.5/4.4/5.7/6.5 µm, p90 21/29/52/71 µm, large-error rate 8.0/7.6/10.6/14.9%. The ≥25% large-error reduction at matched acceptance is **not** met relative to the single-source engine. Every returned event carries position, covariance, region area, flags, accepted flag and reason (`LocalizationResult.to_dict`, Phase 8 schema).

## 8. Decoding revisit (Phase 6) — 8 IBL insertions, 32 units from the most populated region, 5-fold × 4 stratified CV, fixed settings, `reports/phase9/decode_ablation.json`
Mean accuracy: majority 52.9%; counts 59.71%; +electrode 59.72%; +original locations 59.91%; **+P9 locations 59.38%**; +P9 + uncertainty 59.38%; shuffled-position control 59.66%; MLP counts 59.68% vs MLP+P9+uncertainty 59.26%.
Paired (points, bootstrap over sessions): P9 vs counts **−0.33 [−0.85, +0.10]** (3 of 8 sessions positive; H −1.92); P9 vs original locations −0.53 [−1.11, −0.07]; original vs counts +0.20 [−0.09, +0.56]; electrode vs counts +0.02. Target +3 points / CI excluding zero / consistent direction: **not met** (no benefit, consistent with Phase 8).
Why (additional analysis): (a) the spatial features are ≈ deterministic functions of the counts — out-of-fold R² of the position features from counts is 1.0 — so they add no information to a linear decoder; the shuffled-position control matches; (b) the mean-waveform locations are session-global and fixed per unit, so they carry no trial-level signal; localization uncertainty therefore cannot explain the null; (c) accuracy over the majority rate by session is A +16.5, B +19.5, G +7.4, D +4.9, H +3.6, E +3.3, C +0.7, F −1.6 points (the most populated region is often not visual), so only A and B (APN, PO) are clearly informative and region differences cannot be resolved from this sample; (d) the left/right task depends on unit identity (retinotopy), which the counts already encode. Classification accuracy does not measure localization accuracy.

## 9. Runtime and memory
P9 conditioning cost per event relative to corrected B: ×1.2 (amplitude, weighted, robust), ×1.6 (beamform), ×2.9 (template; 0.03→0.16 ms absolute), within the ≤2× target except template. The overlap pipeline is ~7.6 ms/event (Phase 8 full-pipeline class). Memory was not re-measured in Phase 9 (Phase 8's `runtime_bench.json` stands; conditioning adds one copy of a 75×32 snippet).

## 10. Goal status
| Target | Result |
|---|---|
| Corrected baseline reproduced, no unaccounted mismatch, tests added | Met |
| Spread −25% vs best comparable (same units) | **Missed** (−9.5% locked, −17.4% used held-out) |
| Dropout instability −20% | Met for 4/5 known, 5/5 unknown |
| Noise instability −20%, no more outliers | Mostly missed (only amplitude; robust tails worse) |
| Cross-session reproducibility | Mixed: of the 12 locked session×method pairs tabulated (beamform, weighted, robust) 10 improve and 2 are slightly worse (weighted G, H); session F stays poor |
| Synthetic median ≤12 / p90 ≤30 | Met (real-like, model-matched); **fails** on the legacy broad generator |
| Large-error −25% | Met on real-like synthetic; not met for K≤2 pipeline |
| Runtime ≤2× | Met except template (×2.9) |
| Two-source recall ≥90%, false ≤5%, error ≤20 µm | Met (94.5%, 3.2%, 4.0 µm) |
| Confidence rejection reporting 100% | Met |
| Regions ~90% coverage, accepted ≥90% | Approximate (88.6%; 95.6%) |
| Decoding +3 pts | **Missed** (−0.33) |

## 11. Limits and next phase
Locked sessions E–H were used once for P9 (and for the corrected-B/legacy comparators); the used Phase 5 held-out windows are labelled as previously used. The footprint calibration used the development windows only. Real-like synthetic is built from the same measured footprint it is scored with (a model advantage). Triples and dropout-during-overlap were not run. Next: (1) a faithful multi-unit forward model from real templates (the 0.13 relative amplitude at 200 µm in the measured footprint shows structure the monopole/line models miss), (2) real two-unit ground truth, (3) investigate session F, (4) trial-level (not session-global) position features or population-structure decoders instead of adding position columns to counts.

## Reproduce
```sh
PYTHONPATH=. .venv/bin/python scripts/phase9/baseline_corrected.py
PYTHONPATH=. .venv/bin/python scripts/phase9/run_dev_table.py
PYTHONPATH=. .venv/bin/python scripts/phase9/run_final.py          # single use (writes final_locked.json)
PYTHONPATH=. .venv/bin/python scripts/phase9/exp_overlap.py --final  # single use
PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase9_ablation
.venv/bin/python scripts/phase9/summarize.py
.venv/bin/python -m unittest edge.tests.test_phase9 edge.tests.test_phase9_alignment
```
