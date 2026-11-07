# Reports

JSON and markdown here are measured results, not targets. Prefer the files in this table over repeating numbers in the root README.

## Current

| File | What it records |
| --- | --- |
| `setup-validation.json` | MEArec smoke (seed 42) |
| `rtl-pipeline-validation.json` | Verilator vs Python vs MEArec GT, seeded 1 s |
| `rtl-throughput.json`, `rtl-stress.json` | cycle acceptance and dense-stream checks |
| `fixed-point-stress.json` | 100-trial arithmetic stress |
| `formal-verification.json` | decoder + biquad-controller SVA |
| `c5g-fpga-validation.md` | Cyclone V fit, timing, JTAG playback |
| `phase6/phase6-validation.md` | real-AP detector freeze, float vs fixed, locked IBL set |
| `phase6/corrected_history/CORRECTED_PHASE3_4.md` | Phase 3–4 IBL numbers on the probe clock |
| `localization-benchmark-v3.json` | synthetic + MEArec localization |
| `neuropixels-decoder.json`, `synthetic-decoder.json` | Q8.8 count-decoder software/RTL |
| `regression-summary.json` | last `scripts/run_regression.py` |

## Historical (kept, some conclusions superseded)

`phase2-validation.md`, `phase3-validation.md`, `phase4-validation.md`, `phase5/`, `ibl-*.json`, `frozen-real-detector.json`, `localization-benchmark.json`, `localization-benchmark-v2.json`, `edge-decode.json`. See [DEPRECATIONS.md](DEPRECATIONS.md) before quoting Phase 3–5 IBL recall or localization rankings.
