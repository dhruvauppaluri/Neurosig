# Phase 15: autonomous research on localization, overlap, identity, cross-animal decoding and runtime (software only)

Phase 15 pushes the seven research tracks of the Phase 15 brief from the frozen Phase 14 baseline (P14, hash
`9d8006e97d8f`). It changes no SystemVerilog, FPGA, frozen detector, fixed-point or `edge/localize/*` code.
Results: `reports/phase15/phase15-validation.md`. Registry: `reports/phase15/experiment_registry.jsonl`
(`scripts/phase15/log_registry15.py`). Dashboard: `reports/phase15/dashboard_phase15.json`
(`scripts/phase15/build_dashboard15.py`; Phase 14 baseline rows plus Phase 15 rows, same schema, never one combined score).
Frozen configuration: `reports/phase15/frozen_p15.json` (hash `1bcba1777bd6`).

## Data and splits (reserved before any Phase 15 optimization)

| Purpose | Development | Selection | Final (single use) |
|---|---|---|---|
| Biophysical count / position / uncertainty | leave-one-morphology-out over all 13 BBP L5 models (Phases 11-14) | nested inside the same folds | **none**: no never-simulated morphology remains, so Tracks A/B/uncertainty are cross-validated only |
| Real IBL events (identity, real count proxy) | A-C (2 s windows), long 30 s windows A, B | D-S, long D, L, O (all previously used) | X1-X4: 4 new insertions, 4 labs (`reports/phase15/split_manifest_phase15.json`, seed 20271521), 12 x 2 s windows + one 30 s window each |
| IBL visual decoding | Phase 13 protocol (unchanged) | - | X1-X4 |
| Allen cross-animal decoding (`reports/phase15/allen_split15.json`, seed 20271501) | leave-one-mouse-out over 11 Phase 13-14 dev mice | train on the 11, test the 6 Phase 14 final mice (a used benchmark) | 6 never-opened mice; training uses the 17 Phase 13-14 mice + 7 new training mice |

The insertion search (`scripts/phase15/find_insertions15.py`) probed 93 public insertions; one candidate with a
degenerate sync was skipped before any data were read. Allen sessions were split by a seeded permutation before any
was opened; a stalled download was restarted by `scripts/phase15/fetch_watchdog.sh` (resumable, never re-drawn).

## Track A: localization (`edge/phase15/batch.py`, `scripts/phase15/uncertainty15.py`)

A-P15 = the mean of the ms3 and line20 K=1 centres, both fitted on spatially whitened data. The two families err in
different directions on extended neurons (ms3 is biased by the dendritic dipole, line20 by its fixed length), so the
average shortens the error tail. Uncertainty: split-conformal discs calibrated leave-one-morphology-out, Mondrian
bins = observed-SNR tertile x declared overlap.

## Track B: overlap / source count (`edge/phase15/count15.py`, `edge/phase15/trees.py`, `scripts/phase15/train_count15.py`)

B-P15 is a gradient-boosted classifier (scikit-learn HistGradientBoosting, exported to a numpy evaluator so the main
environment needs no scikit-learn) on line20 and ms3 K=1/K=2 statistics, temporal-shape features and observed SNR.
Its threshold is self-calibrated per recording exactly like P14 (quantile of development singles in that recording's
background); the quantile (0.998) was chosen by nested leave-one-morphology-out so the outer folds never see it.
Compared, on the same folds: P14 F21, logistic regression, gradient boosting on all features.

## Track C: online identity (`edge/phase15/identity.py`, `scripts/phase15/run_identity.py`)

ID-P15 = OU-P14 with an amplitude-proportional assignment threshold `max(tau_assign, 1 + tau_rel * mean(mu^2))`,
tau_rel = 0.2: waveform variability grows with amplitude, so a fixed noise-unit tolerance rejected many correct
large-unit events. Tested and rejected: low-SNR likelihood-ratio assignment, look-back re-assignment, refractory,
peak-distance and relative-margin guards, online unit splitting.

## Track D: cross-animal visual decoding (`edge/phase15/xanimal.py`, `scripts/phase15/run_xanimal.py`)

No unit is shared across mice, so features are defined on label-free groups that exist in every mouse: CCF area,
cortical-depth third, waveform width class and functional cluster (k-means of each unit's stimulus-averaged
response time course, centres fitted on training mice only). Feature = per group and 25 ms bin, the mean
response of the group's units, z-scored per session over the evaluation block (label-free, transductive).
Variants: unit normalization, bin width, CORAL alignment, logistic / ridge / MLP / ensemble decoders,
regularization, training-set size, SNR- or quality-weighted pooling (Track E). D-P15 = area x functional cluster,
logistic regression (chosen as best mean lift over chance on dev and select). A per-task routing between D-P15 and
the area x depth MLP was frozen before the final as a secondary analysis.

## Track E: localization -> decoding

Track E tests whether unit quality or position information improves cross-animal decoding: quality- and SNR-weighted
pooling, and depth (a position-derived group). Depth groups helped drifting-grating orientation on dev; weighting
did not help.

## Track F: runtime (`edge/phase15/batch.py`, `scripts/phase15/stream15.py`)

D-P15 batch path: events that share a contact geometry and live mask are processed together; snippets are
compressed to rank 6 in time (separation unchanged, `reports/phase15/tcompress_test.json`), K=1 energies of every
grid centre for every event come from one matrix product per basis family, and the K=2 search uses the Gram
identity of `edge/phase14/fast14.py` for the top 2 first centres. `stream15.py` replays real IBL detector events
through a bounded queue to W worker processes (saturation mode = blocking producer, real-time mode = recorded
inter-event intervals) and reports capacity, end-to-end latency, queue depth and drops; `compute` mode measures
single-process per-event compute latency. Identity association (0.1 ms/event, Phase 14) runs in the collector and
is not part of the measured worker path.

## Reproduce

```bash
data/allen-venv/bin/python scripts/phase15/fetch_allen15.py split && scripts/phase15/fetch_watchdog.sh
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase15/find_insertions15.py && PYTHONPATH=. data/ibl-venv/bin/python scripts/phase15/make_manifest15.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase15/fetch15.py && PYTHONPATH=. data/ibl-venv/bin/python scripts/phase15/fetch_long15.py final
FAM=line20-ms3 .venv/bin/python scripts/phase15/overlap_lomo.py --nested
data/ibl-venv/bin/python scripts/phase15/train_count15.py
.venv/bin/python scripts/phase15/uncertainty15.py
scripts/phase15/batch_xa.sh dev <specs.tsv>        # Track D development; select mode for selection
.venv/bin/python scripts/phase15/run_identity.py dev '{"tau_rel": 0.2}' q_tau_rel0.2
data/phase15/stream_runs.sh                         # runtime (writes reports/phase15/stream15.json)
PYTHONPATH=. .venv/bin/python scripts/phase15/run_final15.py   # single use, refuses to run twice
.venv/bin/python scripts/phase15/log_registry15.py && .venv/bin/python scripts/phase15/build_dashboard15.py
.venv/bin/python -m unittest edge.tests.test_phase15
```

## Limits

Biophysical results come from 13 BBP layer-5 models under MEArec, cross-validated by morphology; there is no
untouched morphology, so they are not final-set results. Real-data identity metrics use sorter labels only for
scoring and are proxies, not ground truth. Cross-animal decoding reads Allen sorted units (not raw voltage).
Runtime is measured on one Apple Silicon laptop under Python/numpy, single-threaded BLAS per worker.
