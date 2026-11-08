# Phase 10: localization gap-closing and a fresh locked evaluation (software only)

Phase 10 runs the parts of the localization brief that Phases 8 and 9 did not cover. It does not repeat their
experiments. Phase 9 had already spent the Phase 6 locked sessions E–H, so Phase 10 reserves four new insertions
(I–L) and evaluates them once. No RTL, FPGA, fixed-point detector, or `edge/localize/*` file changed. No new
beamforming family was added.

## What is new

| Area | Module | Notes |
|---|---|---|
| Recording-segment stability, SNR dependence, peak-electrode removal | `edge/phase10/realextra.py` | Per session, windows are ranked by start time and split into early, middle and late tertiles. SNR is the sorter-channel trough depth over edge noise, after baseline subtraction. The peak contact is the deepest trough found in the data. Peak removal is tested two ways: known (engine rebuilt) and unknown (contact reads zero). A null set (+20 ms) gives the metric floor. |
| Overlap stress | `edge/phase10/overlap_stress.py` | Sweeps of temporal offset (\|dt\| 0 / 1–3 / 4–9 / 10–20 samples) and separation. Three-source events. Dropout during overlap. Single-source negative controls. Seed pools `p10dev`, `p10test` and `p10confirm` are new. The resolvable rule is the Phase 9 one. Calibration is the frozen Phase 9 calibration. |
| Greedy K=3 extension (opt-in) | `k3_fit` | Adds a third source at the best grid cell orthogonal to the accepted K=2 span, then refines all three. The threshold is set on `p10dev` only, as the 97th percentile of delta3 on resolvable pairs. |
| Dead-contact exclusion | `detected_dead_mask`, `edge/phase10/configs.py` | Contacts that the P9 rule detects as flat are excluded instead of interpolated. The overlap fitter receives them as a mask. For the single-source engines (P10), the P9 engine is rebuilt on the live contacts, with an LRU cache of 16 masks. With no dead contact, P10 output is identical to P9. |
| Fresh locked set | `scripts/phase10/{find_insertions,make_manifest10,fetch10,run_final10}.py` | Four insertions, never used before, from four labs. Each has at least 32 VIS contacts and raw AP. Six 2 s windows per session, seed 20271106. Two candidates with degenerate (2-row) probe sync files were skipped for availability before any window was read. |
| Decoding on new insertions | `edge/decode/phase10_new_sessions.py` | The frozen Phase 9 protocol, plus an MLP shuffled-position control with identical capacity. |
| Runtime and memory | `scripts/phase10/runtime_memory.py` | Wall and CPU latency, tracemalloc peak and resident engine size, and the cost of the P10 rebuild path. |

## Configurations

- **P9** (hash `5ed8a69b97eb`): frozen in Phase 9 and unchanged.
- **P10** (hash `8a56becbfc85`): P9 plus dead-contact exclusion. It was frozen from development windows A–D and
  synthetic pools before the I–L windows were downloaded.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/phase10/real_extra_dev.py dev
PYTHONPATH=. .venv/bin/python scripts/phase10/overlap_stress.py          # single-use p10test pool (refuses rerun)
PYTHONPATH=. .venv/bin/python scripts/phase10/exp_dead_mask.py           # p10dev; --confirm runs p10confirm once
PYTHONPATH=. .venv/bin/python scripts/phase10/exp_x2_dev.py
PYTHONPATH=. .venv/bin/python scripts/phase10/runtime_memory.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase10/fetch10.py
PYTHONPATH=. .venv/bin/python scripts/phase10/run_final10.py             # single-use, refuses rerun
PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase10_new_sessions
PYTHONPATH=. .venv/bin/python scripts/phase10/log_registry.py
.venv/bin/python -m unittest edge.tests.test_phase10
```

Results and limits are in `reports/phase10/phase10-validation.md`.
