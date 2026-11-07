# Neurosig agent guidance

Reproducible path: MEArec → software reference → SystemVerilog detector → decoder inference → optional FPGA / later SKY130. Neuropixels is the inspiration; this repo does not implement a probe or analog front end. Default channel count is 32; keep it configurable.

## Frozen contracts

There are two detectors. Mixing them is the usual way to report a wrong number.

1. **MEArec / current RTL / C5G** — 32 kHz, no CAR, 300 Hz–6 kHz IIR, negative 6σ. Golden models: `scripts/filter_iir.py`, `scripts/detect_spikes.py`, `scripts/fixed_point_model.py`. RTL: `rtl/neurosig_pipeline.sv`. Do not change rounding or saturation without rechecking poles and the zero-input decay test (`scripts/stress_fixed_point.py`). Section outputs truncate toward zero.
2. **Real AP, software-only** — 30 kHz, `detector-v2.0` + integer `fixed-v1.1` (200–4500 Hz, median CAR, 4σ). Spec: `docs/phase6-rtl-contract.md`. Compare new arithmetic to `scripts/phase6/oracles.py`, not the model against itself. Bump the version, regenerate `tests/vectors/phase6/`, and rerun float-vs-fixed if the math changes. **No RTL implements this yet.**

Use `.venv/bin/python` after `bash scripts/setup.sh`. Locks in `requirements*.txt` are the package source of truth. Keep venvs, MEArec outputs, IBL caches, and FPGA build products out of Git.

## Do not

- Edit `rtl/` or golden-model arithmetic to support `viz/`. Viewer code stays under `viz/`. Neuropixels playback is sorted spikes, not ADC samples.
- Fold localization into `scripts/` DSP. It lives in `edge/`. `viz/sim/localization.py` is a display adapter. No localization RTL until a bit-accurate Python model and packet freeze exist (`docs/rtl-acceleration-localization.md`).
- Compare IBL sorter times to raw samples without `scripts/phase5/data.py` (session clock → probe samples). Quote channel-consistent F1 plus the shifted-reference chance level, not unconstrained recall alone.
- Use ground-truth xyz as decoder features. Occupancy bins must use the recording’s y range, not the synthetic 0–620 µm default.
- Treat missing IBL windows as passes; `scripts/run_regression.py` reports `missing_dataset`.
- Claim C5G IBL playback as the CAR/4σ detector. That board path is the no-CAR 32 kHz adaptation.
- Report area, power, or clock as measured unless the tool actually ran. The 100 MHz figure on the generic pipeline is an assumption; C5G is a 50 MHz fit of a different schedule.

## Tests

```sh
.venv/bin/python scripts/run_regression.py
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s edge/tests
```

Formal: `bash formal/run_formal.sh`. FPGA: `fpga/README.md`. Current vs superseded reports: `reports/README.md`.

## Open

- Implement `fixed-v1.1` in RTL against the Phase 6 vectors.
- SKY130 / OpenROAD after that match.
- Visualization markers for truth vs estimate vs electrodes must stay distinct. CCF is not a human neuron coordinate.

If you explain a conversion or threshold, let the user compute the quantization before pasting a finished implementation. Record seeds, widths, and the command that produced a claimed number.
