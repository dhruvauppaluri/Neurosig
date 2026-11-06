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
