# Deprecated conclusions (original reports are kept unchanged)

The files listed here are **not deleted or edited**; they remain as historical records. The conclusions named below are superseded.

| Original | Superseded statement | Reason | Replacement |
|---|---|---|---|
| `reports/ibl-detection-localization.json`, `reports/phase3-validation.md`, `reports/phase4-validation.md` | Recall of the improved IBL detector "3.4 % → 49.0 %" (first insertion) and "8.1 % → 46.9 %" (second); per-window TP/FP/FN | Sorter times (session clock) were compared with raw samples (probe clock); offsets up to −31 ms on later windows. Unconstrained 1 ms recall also contains a large chance component | `reports/phase6/corrected_history/CORRECTED_PHASE3_4.md` |
| `reports/ibl-pilot.json`, localization sections of `reports/ibl-detection-localization.json` | Method ranking by median axial disagreement (amplitude 41, template 80, beamform 62, robust scan 42 µm) | Snippets were centred 1.5 ms off the true spike peak | corrected values 22 / 20 / 19 / 23 µm (same file) |
| `reports/ibl-detector-decode.json` | Detector-event vs sorted-count decode on 40 s blocks | Probe-clock detector events mixed with session-clock `stimOn_times` (window shifted 17-21 ms of 200 ms); 10-11 trials, already inconclusive | not re-run (see corrected report) |
| `reports/phase5/phase5-validation.md` | "98.4 % of float events identical" for `fixed-v1`; `docs/phase5-fixed-point-spec.md` constants | metric is exact-match/float (one-sided; Jaccard 96.9-97.0 %); constants 97157/582964 were arithmetic slips; no s24 input saturation | `docs/phase6-rtl-contract.md` (fixed-v1.1), `reports/phase6/mismatch-classification.md` |
| `reports/phase5/phase5-validation.md` | Phase 5 held-out windows as an untouched final holdout | They were evaluated and discussed; they are now a previously-used benchmark | `reports/phase6/generalization.md` |

Unaffected: `reports/ibl-decode.json` (uses only session-clock quantities), MEArec/RTL results, formal verification, synthetic benchmarks.

## Phase 9 additions (localization stability)
* **Raw-snippet within-unit spreads are not evidence of localization quality.** A null control (same units' times shifted +20 ms) shows raw-path spreads (e.g. amplitude 7.4 µm real vs 6.6 µm null) are static channel-offset artifacts. Any stability number computed without baseline subtraction and a null comparator, including the Phase 4 pilot's within-unit consistency and Phase 8 `reports/phase8/real_stability_*.json` (different filtered/re-aligned path, no null control), should be read as unvalidated. Originals are kept. See `reports/phase9/phase9-validation.md` §2.
* **Phase 8 overlap results do not transfer** to a real-like forward model with shared noise (resolvable recall 18.5% as configured); Phase 9's nuisance-common-mode variant replaces the CAR-conditioned configuration.
* **Phase 8 synthetic accuracy** came from a broad-footprint generator (λ=80 µm); the measured real footprint is ~λ=20 µm. Those synthetic gains should not be used to rank methods on real recordings.

## Phase 10 additions
* **Phase 9 overlap false two-source rate (3.2%)** was optimistic. Three fresh real-like pools give 7.5-8.3% with the same frozen calibration (`reports/phase10/overlap_stress.json`, `exp_dead_mask_*.json`). Phase 9's resolvable-pair recall (94.5%) measured 89.3% on the fresh p10test pool.
* **Phase 9 dead-contact interpolation in the overlap pipeline** produces phantom second sources under unknown dropout (52-87% false two-source); use the Phase 10 mask (`dead_as_mask=True`).

## Phase 11 additions
* **False two-source rates from single pools are not stable.** Five fresh 120-single pools (Phases 10-11) gave 3.3-9.2% with similar rules (`reports/phase11/final11.json`, `exp11e_overlap.json`). Single-pool figures, including Phase 9's 3.2% and Phase 10's 7.5%, should be read with that spread.
* **Phase 9 conformal regions** under-cover low-SNR events (78% at nominal 90%) and are oversized for pairs (median area 6,668 um^2 on p11select). Superseded by the Phase 11 Mondrian calibration (`reports/phase11/calibration_p11.json`).
