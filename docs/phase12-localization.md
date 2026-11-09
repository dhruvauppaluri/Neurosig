# Phase 12: unit-level localization, spatiotemporal fitting, source counting, uncertainty and runtime (software only)

Phase 12 separates **event-level** localization (where did this event come from?) from **unit-level** localization (where
is the neuron behind many events?). It tests spatiotemporal amplitude extraction, model-mismatched (biophysical)
validation, source counting on realistic waveforms, uncertainty over the complete event population, and real-stream
throughput. No RTL, FPGA, fixed-point detector, frozen Phase 6 detector or `edge/localize/*` file changed.
Results are in `reports/phase12/phase12-validation.md`; the frozen configuration is **P12** (`edge/phase12/configs.py`).

## Data and splits (session-disjoint)

| Split | Sessions | Use |
|---|---|---|
| dev | A-C (Phase 5 calibration + selection windows) | every decision |
| select | D (Phase 5 held-out), E-H, I-L, M-P | previously used sessions; checks and model selection only |
| final | **Q-S** (`reports/phase12/split_manifest_phase12.json`, 8 x 2 s windows each) | evaluated once by `scripts/phase12/run_final12.py` |

Only three fresh insertions met the eligibility rules (>= 32 visual contacts, raw AP, non-degenerate probe sync,
distinct labs); the fourth candidate lab had a degenerate sync file. This is an availability limit, recorded in the
manifest.

Synthetic pools p12dev / p12select / p12final (seeds 20271401-3) combine the Phase 9/10 line-source generator with a
**model-mismatched biophysical family**: MEArec/NEURON templates (6 L5 cell models) plus real IBL background noise
(`edge/phase12/biosim.py`). Dev and select use disjoint halves of the Phase 11 template set; final uses a fresh
template set (seed 20271308) and noise from Q-S.

## Modules

| Module | Purpose |
|---|---|
| `edge/phase12/pooling.py` | Session event store (raw segments, E3 shifts, SNR, detector match); six unit-level estimators (single, mean, median, Huber-robust, SNR-weighted geometric median of positions, joint line20 with per-spike amplitudes) on disjoint pools of N spikes; pairwise disagreement and temporal drift |
| `edge/phase12/online_groups.py` | Sorter-free conservative online grouping of detector events (local waveform correlation >= 0.9) |
| `edge/phase12/spatiotemporal.py` | Single-event amplitude variants (trough, mean3, rank-1, matched filter, GLS with spatial noise covariance, adaptive contacts, MAP free-width line source, high-pass versions) and the identifiability area |
| `edge/phase12/biosim.py` | Biophysical templates + real noise simulator; template pools; axial template moves for pairs / triples |
| `edge/phase12/overlap_bench.py`, `count_model.py` | Expanded single / pair / triple benchmark; temporal and second-source count features; P11 and P12-LR decisions; cheap gate |
| `edge/phase12/dictionary.py` | Empirical spatiotemporal waveform dictionary as a single-neuron null (rejected) |
| `edge/phase12/fast.py` | FastLine20 (analytic trust-region Gauss-Newton), E3 / mean3 helpers, analytic-Jacobian K<=2 refine with closed-form shared eigenvalue |
| `edge/phase12/configs.py` | Frozen P12 (hash in reports) |

## Frozen P12

* Single event: E3 alignment, mean of 3 samples around the trough, line20 fit (FastLine20); uncertainty = disc of area
  c(SNR group) x identifiability area (`reports/phase12/exp12_5_real_proxy.json`).
* Unit level: Huber-robust mean of E3 trough amplitudes + line20 (`l20_robust`), reported at N = 5/10/25/50.
* Sorter-free grouping: rho 0.9 (purity first).
* Source count: K<=2 space-time analysis (fast refine) + logistic rule `reports/phase12/count_model_p12.json`; no gate;
  K = 3 off.
* Multi-source regions: Mondrian over count x SNR x chi2 tercile x dropout, fitted on all p12dev events
  (`reports/phase12/calibration_p12.json`); unbounded regions are reported as such.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/phase12/reproduce_baselines.py
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_2.py dev --population matched     # also select, --population all
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_2b.py dev 0.9                     # also 0.8, 0.7; select 0.9
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_3.py dev                          # also select
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_3_biophys.py dev                  # also select
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4.py build p12dev && PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4.py fit
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4.py build p12select && PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4.py eval p12select
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4c.py bio p12dev                  # dictionary null (rejected); also p12select
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_4_real.py select
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_5.py && PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_5_real.py
PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_6.py gate && PYTHONPATH=. .venv/bin/python scripts/phase12/run_12_6.py stream select 40
.venv/bin/python scripts/phase12/gen_biophys_final.py                                  # final-pool templates
PYTHONPATH=. .venv/bin/python scripts/phase12/run_final12.py                           # single use
PYTHONPATH=. .venv/bin/python scripts/phase12/log_registry.py
PYTHONPATH=. .venv/bin/python -m unittest edge.tests.test_phase12
```

`data/ibl-venv` is needed for `make_manifest12.py`, `fetch12.py` and `fetch_cluster_info.py`. Session caches, noise
banks, benchmark pickles and templates are git-ignored under `data/phase12/`.
