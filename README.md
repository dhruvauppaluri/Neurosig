# Neurosig

Hardware-oriented spike detection for Neuropixels-style recordings: simulate a short MEArec stream, keep a bit-accurate software reference, implement the filter and detector in SystemVerilog, and check it in simulation and on a Cyclone V board. Localization, packet emulation, and a 3D viewer sit beside that path; they are not on-chip.

There are two detector contracts. Do not mix them.

| | MEArec / current RTL / C5G | Real AP (Python, frozen for later RTL) |
| --- | --- | --- |
| Rate | 32 kHz | 30 kHz native |
| Front end | per-channel 300 Hz–6 kHz IIR, no CAR | 200–4500 Hz IIR, then median CAR |
| Threshold | negative 6σ MAD | 4σ, offline registers |
| Status | Verilator + C5G match Python on the seeded 1 s recording | `fixed-v1.1` / `detector-v2.0` specified and vectorized; **no RTL yet** |

Sorter output is an agreement reference, not biological ground truth. SKY130 / OpenROAD work has not started.

## Layout

| Path | Role |
| --- | --- |
| `scripts/` | MEArec smoke, ADC, float/fixed DSP, regression |
| `rtl/` | SystemVerilog filter, detector, merge, count decoder |
| `fpga/` | Cyclone V GX (C5G) JTAG playback |
| `formal/` | SVA k-induction for the decoder and biquad controller |
| `edge/` | localization, NSG1 software link, IBL pilots |
| `viz/` | research UI (sorted-spike playback, not ADC) |
| `docs/` | frozen contracts and interface notes |
| `reports/` | measured results; see [reports/README.md](reports/README.md) |
| `tests/` | unit tests and Phase 6 RTL vectors |

## Setup

Needs network, Python 3, a C/C++ compiler, and `make`.

```sh
bash scripts/setup.sh
.venv/bin/python scripts/smoke_mearec.py
.venv/bin/python scripts/run_regression.py
```

`setup.sh` installs a project-local `uv` under `.tools/` and the env under `.venv/`. Locks live in `requirements-lock.txt`. Generated recordings, caches, and venvs stay out of Git.

The smoke test (seed 42) writes `data/smoke/`: two templates, 32,000 samples × 32 channels at 32 kHz, nine ground-truth spikes. [MEArec](https://mearec.readthedocs.io/en/latest/installation.html).

## Software reference (MEArec path)

```sh
.venv/bin/python scripts/unfiltered_mearec.py   # optional: inspect range before ADC
.venv/bin/python scripts/quantize_adc.py        # signed 16-bit, ±135 µV
.venv/bin/python scripts/filter_iir.py          # 300 Hz HP + 6 kHz LP biquads
.venv/bin/python scripts/detect_spikes.py       # 6σ, 1 ms refractory, 0.5 ms merge
.venv/bin/python scripts/fixed_point_model.py   # Q2.30 coeffs, 24-bit states
.venv/bin/python scripts/stress_fixed_point.py --trials 100
```

ADC: one LSB = `270/65536` µV; round half away from zero, then saturate. Filter sections truncate toward zero (nearest rounding left a 127-code zero-input limit cycle). Thresholds are offline MAD, not on-chip. Numbers for the seeded recording and the 100-trial stress suite are in `reports/setup-validation.json`, `data/detections/summary.json`, and `reports/fixed-point-stress.json`.

## RTL and formal

The pipeline is `sample_input` → two biquads → per-channel detector → 16-sample merge (`rtl/neurosig_pipeline.sv`). Directed testbenches live next to each module under `rtl/tb/`. Full-recording check:

```sh
.venv/bin/python rtl/tb/generate_pipeline_reference.py
verilator --binary --timing -j 4 -Wno-fatal --top-module neurosig_pipeline_reference_tb \
  --Mdir /tmp/neurosig-verilator-pipeline \
  rtl/tb/neurosig_pipeline_reference_tb.sv rtl/neurosig_pipeline.sv rtl/sample_input.sv \
  rtl/spike_band_filter.sv rtl/biquad_controller.sv rtl/biquad_controller_pipelined.sv \
  rtl/biquad_state.sv rtl/biquad_arithmetic.sv rtl/spike_detector_multichannel.sv \
  rtl/spike_detector_channel.sv rtl/spike_event_merge.sv rtl/spike_event_merge_serial.sv
/tmp/neurosig-verilator-pipeline/Vneurosig_pipeline_reference_tb \
  +ADC=data/quantized/adc_codes.hex +THRESHOLDS=/tmp/neurosig-thresholds.txt \
  +EVENTS=/tmp/neurosig-merged-events.txt +OBSERVED=/tmp/neurosig-rtl-events.txt
.venv/bin/python rtl/tb/evaluate_pipeline_events.py
```

That run matches all nine Python events on the seeded recording (`reports/rtl-pipeline-validation.json`). Throughput and dense-stream stress: `reports/rtl-throughput.json`, `reports/rtl-stress.json`. Assert `flush` only after the filter drains; reset between recordings.

Count decoder: `scripts/synthetic_decoder.py` and `scripts/neuropixels_decoder.py` train Q8.8 linear models; `rtl/spike_count_decoder.sv` matches those scores. The Neuropixels file is sorted trial counts, not raw voltage.

SVA proofs for the decoder score and the biquad handshake: [`formal/README.md`](formal/README.md). The controller proof black-boxes arithmetic.

## FPGA

[`fpga/README.md`](fpga/README.md) — Terasic C5G, 50 MHz, JTAG mailbox. Seeded 1,024,000-sample playback matched the nine Python/Verilator events. Host transfer is far below real time. A public 1 s IBL window, resampled 30→32 kHz, matched the **no-CAR** Python event; that is not the frozen CAR detector. Record: [`reports/c5g-fpga-validation.md`](reports/c5g-fpga-validation.md).

## Real recordings (Python)

IBL AP is 30 kHz. Sorter times must be mapped onto the probe clock (`scripts/phase5/data.py`). Chance-adjusted, channel-consistent F1 is the metric to quote; unconstrained 1 ms recall is inflated.

- Frozen software model: `detector-v2.0` + `fixed-v1.1` (median CAR). Contract: [`docs/phase6-rtl-contract.md`](docs/phase6-rtl-contract.md). Vectors: `tests/vectors/phase6/`.
- Engineering summary: [`reports/phase6/phase6-validation.md`](reports/phase6/phase6-validation.md).
- Older Phase 3–4 IBL numbers used the wrong clock; originals kept, conclusions listed in [`reports/DEPRECATIONS.md`](reports/DEPRECATIONS.md).

Optional IBL env: `./scripts/setup_ibl.sh` then `data/ibl-venv/bin/python …`. Cache stays under `data/neuropixels/`.

## Edge and viewer

`edge/` localizes snippets (amplitude, template, quasi-static scan). `robust_scan` refuses a point if the trough footprint is bimodal. Template matching is the default recommendation on the synthetic set. Localization RTL is not implemented.

`viz/` is a separate UI: stylized anatomy, toy synthetic activity, and mouse sorted-spike playback on the visual region. Setup: [`viz/README.md`](viz/README.md).

## Next

1. Implement `fixed-v1.1` in SystemVerilog against the Phase 6 vectors.
2. Decide whether 200–4500 Hz stays the shipped band (one of four locked IBL sessions got worse).
3. SKY130 / OpenROAD only after that RTL matches the vectors.

When changing arithmetic, record seed, widths, rounding, and a command that reproduces the number. Agent rules: [`AGENTS.md`](AGENTS.md).
