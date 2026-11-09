# Phase 14: neural source identity, overlap resolution, online localization and visual decoding (software only)

Phase 14 asks first whether one complex neuron can be told apart from two overlapping neurons, then whether units
can be associated and tracked online without a sorter, and finally which temporal and spatial features carry visual
information across recordings and animals. No SystemVerilog, FPGA, frozen detector or fixed-point code changed.
Results: `reports/phase14/phase14-validation.md`. Registry: `reports/phase14/experiment_registry.jsonl`.

## Data and splits (reserved before any Phase 14 evaluation)

| Purpose | Development | Selection | Final (single use) |
|---|---|---|---|
| Biophysical count / position / uncertainty (`edge/phase14/bench14.py`) | p14dev: TTPC1, LBC, MC (Phase 11/12 templates), IBL A-C noise | p14select: UTPC, STPC, NBC, IBL E-H noise; p14sel_np2: same models on Neuropixels 2.0 | p14final: 7 never-simulated BBP L5 models (TTPC2, BP, BTC, ChC, DBC, NGC, SBC) on NP1 (84 templates) and NP2 (56), noise from fresh IBL T-W |
| Drift streams (`edge/phase14/drift_sim.py`) | dev morphologies | select morphologies | unseen morphologies |
| Real IBL events (online association, real count check) | A-C | D-S (all previously used) | T-W: 4 new insertions from 4 labs (`reports/phase14/split_manifest_phase14.json`, 12 x 2 s windows each) |
| IBL visual decoding | Phase 13 dev/select/final (historical) | - | T-W (Phase 13 protocol, unchanged) |
| Allen visual decoding (`reports/phase14/allen_split14.json`) | 2 Phase 13 dev mice | 3 Phase 13 mice + 6 new mice | 6 new mice (seeded draw before any session was opened) |

Template sets: `scripts/phase14/gen_templates14.py` (seeds 20271411-13). Morphology splits are disjoint and tested.

## 14.2-14.3 Single-neuron null and source count

`edge/phase14/neuron_model.py`: a source at in-plane centre p spans a small spatial basis with one free temporal
waveform per column. Families: `line20` (Phase 9-12), `ms` (g15, g40, g100, axial dipole), `ms5`, `msq`, `msn`, `msw`,
`ms3` (g15, g40, axial dipole). Count statistic F21 = per-parameter K=2 energy gain over the K=2 residual.
Noise: per-contact (diagonal), spatial (`NoiseModel`, background covariance) or separable spatio-temporal whitening.
Thresholds are self-calibrated per recording without labels: the 98th percentile of F21 over development-morphology
MEArec singles placed in that recording's own event-free background. K=3: F32 > 95th percentile of F32 over
development pairs. The overlap probability is a logistic map of log F21 (balanced fit on development) converted to a
stated prior.

Frozen P14 (`reports/phase14/frozen_p14.json`, hash `9d8006e97d8f`): `ms3` with spatial whitening, 2% self-calibrated
threshold, fast search (2 first-centre candidates, 1 refinement round), K=1 position = ms3 centre, conformal disc
uncertainty (radius 9.83 x identifiability scale, dev 90%).

## 14.4-14.5 Online association and drift

`edge/phase14/online_units.py` (stages A-F in the module docstring). Distances are noise-normalized waveform residuals
over shifts and bounded gains on each unit's own contact set; events are assigned only with an energy gain >= 80
(noise units^2) over the null and SNR >= 6; ambiguous events are left unassigned; tentative units need 8 events.
Prototype merging and silent-unit relinking were rejected (see registry). Sorter labels are used only by
`evaluate()`. Drift trackers (`drift_sim.track_positions`): static, window, EMA, Kalman, Kalman + CUSUM change point;
frozen: Kalman, q = 2 um/sqrt(s), r = 15 um.

## 14.7 Runtime

`edge/phase14/fast14.py` reproduces the ms3 statistic exactly (regression test) with cached whitened grids, a
Gram-matrix projection identity in place of per-candidate QR, and a lazily filled fine grid.

## 14V Visual decoding

`scripts/phase14/fetch_allen14.py` fixes the Phase 13 Allen waveform map (`edge/phase14/allen_map.py`; rows follow
`local_index`, regression-tested) and stores 5 ms bins. `scripts/phase14/fetch_gabor14.py` adds receptive-field Gabor
positions. `scripts/phase14/run_vis14.py` documents the protocol (fixed on dev). `scripts/phase14/run_ibl14.py`
reapplies the frozen Phase 13 IBL protocol to T-W.

## Reproduce

```bash
.venv/bin/python scripts/phase14/gen_templates14.py final_np1      # also final_np2, sel_np2
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/find_insertions14.py && PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/make_manifest14.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/fetch14.py
data/allen-venv/bin/python scripts/phase14/fetch_allen14.py split && data/allen-venv/bin/python scripts/phase14/fetch_allen14.py fetch 6
data/allen-venv/bin/python scripts/phase14/fetch_gabor14.py
.venv/bin/python scripts/phase14/run_14_2.py p14dev       # p14select, p14sel_np2, cal:<pool>
.venv/bin/python scripts/phase14/select_14_3.py && .venv/bin/python scripts/phase14/run_14_3_ident.py
.venv/bin/python scripts/phase14/run_p14.py p14dev 2 1 && .venv/bin/python scripts/phase14/eval_p14.py calibrate _c2r1
.venv/bin/python scripts/phase14/build_det_cache.py && .venv/bin/python scripts/phase14/sweep_14_4.py 1
.venv/bin/python scripts/phase14/run_14_5.py p14dev 3 && .venv/bin/python scripts/phase14/run_14_3_real.py select
.venv/bin/python scripts/phase14/run_final14.py            # single use
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/run_ibl14.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase14/run_vis14.py within select   # cross select, within final, cross final
.venv/bin/python scripts/phase14/runtime14.py && .venv/bin/python scripts/phase14/log_registry14.py
```
