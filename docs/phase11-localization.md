# Phase 11: real-neuron localization reliability (software only)

Phase 11 asks why real Neuropixels localization stays unstable, and tests fixes in the order the brief sets out:
event referencing first (a hard checkpoint), then forward models, difficult sessions, dead-contact tails,
source-count selection, conditional calibration and runtime. No RTL, FPGA, fixed-point detector, frozen Phase 6
detector or `edge/localize/*` file changed. Results are in `reports/phase11/phase11-validation.md`.

## Modules

| Module | Purpose |
|---|---|
| `edge/phase11/referencing.py` | Strategies A (sorter), B (frozen detector-v2.0 float reference), C (sorter time with a common multichannel shift, ±0.2 ms) and E3 (the same rule on the engines' own raw-CAR signal). Detector–sorter matching uses optimal one-to-one assignment at ±0.5 ms within 60 µm, with a shifted-reference chance level. SNR and trough measures use the frozen preprocessing. |
| `edge/phase11/exp11a.py`, `decompose.py` | Localization of every event and its +20 ms null; matched-event and full-stream summaries; timing and SNR perturbation curves. |
| `edge/phase11/forward_models.py`, `footprint.py` | Six forward models: line20 (baseline), linefree, empirical radial, empirical anisotropic, twoline, and line20 + offset. Variable-projection fitter with cached vectorized grids. Footprint statistics; leave-one-session-out held-out-contact prediction. |
| `edge/phase11/dropout.py` | Interpolate-or-exclude features and fallback policies for dead contacts. |
| `edge/phase11/overlap.py` | Source-count rule variants: axial-only boundary veto (OV-A), misfit-normalized statistics (OV-B), gated K=3. |
| `edge/phase11/calibration.py` | Mondrian split-conformal over stratum × SNR group × dropout, conservative pooling, Wilson intervals, risk–coverage. |
| `edge/phase11/staged.py` | Staged runtime with two ambiguity gates; both were rejected. |
| `edge/phase11/configs.py` | Frozen **P11** (hash `188c8efd636e`): P10 engines, mask cache of 4, the 11D fallback, a weak-trough flag, OV-A, and Mondrian calibration (`reports/phase11/calibration_p11.json`). |

## Data

| Split | Windows | Use |
|---|---|---|
| dev | Phase 5 calibration + selection, sessions A–C | Development and decisions |
| selection | Phase 5 held-out + Phase 6 E–H + Phase 10 I–L | Previously used; model selection only |
| final | **M–P**, `reports/phase11/split_manifest_phase11.json` | Four new insertions, evaluated once by `scripts/phase11/run_final11.py` |

Synthetic pools p11dev, p11select and p11final use seeds 20271201–3. The biophysical templates
(`scripts/phase11/gen_biophys_templates.py`, MEArec/NEURON, seed 20271108) are git-ignored under `data/phase11/`.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11a.py dev
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11a.py dev --align-half 3 --align-mode raw
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11a.py select
PYTHONPATH=. .venv/bin/python scripts/phase11/decompose_11a.py
.venv/bin/python scripts/phase11/gen_biophys_templates.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11b.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11b.py --models line20,line20c --out reports/phase11/exp11b_dev_line20c.json
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11b_biophys.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11b_select.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase11/fetch_cluster_metrics.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11c.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11d.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11e.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11f.py
PYTHONPATH=. .venv/bin/python scripts/phase11/run_11g.py --mode greedy2
PYTHONPATH=. .venv/bin/python scripts/phase11/run_final11.py     # single use, refuses rerun
PYTHONPATH=. .venv/bin/python scripts/phase11/log_registry.py
.venv/bin/python -m unittest edge.tests.test_phase11
```

`run_11c.py` reads `data/phase11/cluster_metrics_plain.pkl`. `.venv` has no pandas, so convert the output of
`fetch_cluster_metrics.py` to that plain-list pickle with `data/ibl-venv` first.
