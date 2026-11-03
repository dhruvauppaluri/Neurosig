# Neurosig

Neurosig is a neural signal processing hardware project inspired by Neuropixels recordings. The plan is to generate synthetic extracellular recordings, implement spike detection in SystemVerilog, train a small neural decoder, and implement its inference in RTL. A later milestone is to evaluate the digital design using SKY130 and an open source physical design flow.

## Current status

The MEArec setup is working on Apple Silicon macOS with Python 3.12.15, MEArec 1.11.0, NEURON 8.2.7, and LFPy 2.3.6. The verified smoke test compiles NEURON mechanisms, generates two biophysical templates, and produces a one-second, 32-channel recording at 32 kHz with nine ground-truth spikes (seed 42). It reloads the HDF5 recording and checks its shape, finite nonconstant samples, and spike ground truth. See [the validation summary](reports/setup-validation.json).

The first recording target is **32 channels**. The smoke test uses one excitatory neuron and a Neuronexus-32 probe. The floating-point and fixed-point software references and a complete RTL filter and spike detector pipeline are implemented. The seeded one-second end-to-end RTL test matches all nine Python merged events exactly; decoder selection is the next project milestone.

A separate research UI under [`viz/`](viz/README.md) shows a translucent anatomical human brain surface, illustrative synthetic activity and wiring, and fixture Neuropixels **sorted-spike** playback confined to the visual region. It imports the golden-model Python APIs and does not modify RTL.

## Setup and validation

Run these commands from the repository root:

```sh
bash scripts/setup.sh
.venv/bin/python scripts/smoke_mearec.py
```

Setup requires network access and an existing Python 3 interpreter to bootstrap `uv`. Template generation requires a working C/C++ compiler and `make` (Xcode Command Line Tools on macOS). The current lock was verified on Apple Silicon macOS.

The setup uses a project-local `uv` executable under `.tools/` and a Python environment under `.venv/`. Dependency declarations live in `requirements.txt`; exact resolved versions are recorded in `requirements-lock.txt`. The smoke test writes generated artifacts under `data/smoke/`. These generated files and local environments should stay out of Git.

Outputs are `templates.h5`, `recording.h5` (including spike ground truth), `parameters.yaml`, and `summary.json`. The smoke script uses a temporary path alias for MEArec and NEURON subprocesses so this checkout can live in a folder containing spaces. It reads the packaged defaults directly and keeps generated models and compiler output temporary.

To inspect the voltage range before quantization, run `.venv/bin/python scripts/unfiltered_mearec.py` after the smoke test. This reuses its templates, seeds, and neuron selection with the recording band-pass filter disabled. It writes `data/unfiltered/recording.h5`, `parameters.yaml`, and `traces.png`; the original smoke recording stays available. The script verifies the saved recording shape, finite samples, disabled filter, and unchanged spike times. These are simulated extracellular voltages with added noise; disabling the recording filter does not model a complete analog front end.

Quantize the unfiltered recording with:

```sh
.venv/bin/python scripts/quantize_adc.py
```

The current reference uses a signed 16-bit two's-complement format over a symmetric ±135 µV input range. One LSB is `270 / 65536 = 0.004119873046875 µV`. It rounds to the nearest code with exact halves going away from zero, then saturates to −32,768…32,767. The positive endpoint is exclusive: the greatest representable voltage is one LSB below +135 µV.

Generated files under `data/quantized/` include NumPy codes, little-endian signed binary, one four-digit hexadecimal two's-complement code per line, and a JSON measurement report. Values are flattened in sample-major/channel-minor order: all 32 channels for sample 0, then all 32 channels for sample 1. These generated outputs remain outside Git.

Build the floating-point IIR reference with:

```sh
.venv/bin/python scripts/filter_iir.py
```

The current filter is causal and hardware-oriented: a second-order Butterworth high-pass at 300 Hz followed by a second-order Butterworth low-pass at 6 kHz. Each stage is stored as a biquad (second-order section). The program checks that every floating-point pole magnitude is below one, filters all channels independently from zero reset state, and writes coefficients, filtered samples, response plots, and a stability report under `data/filtered/`. The fixed-point arithmetic reference is described below.

Run the floating-point spike detector with:

```sh
.venv/bin/python scripts/detect_spikes.py
```

It estimates each channel's noise with the median absolute deviation, uses a configurable negative six-sigma threshold, timestamps the local minimum after each crossing, applies a 1 ms channel refractory period, and merges detections across channels within 0.5 ms. It compares the merged events with MEArec ground truth using a 1 ms tolerance. Outputs under `data/detections/` include an event CSV, JSON metrics, and a detection plot. The default achieved 9/9 true positives with no false positives on the seeded one-second smoke recording; this small result is pipeline validation rather than evidence of general detector performance.

Run the bit-accurate fixed-point reference with:

```sh
.venv/bin/python scripts/fixed_point_model.py
```

The model uses signed Q2.30 biquad coefficients, signed 24-bit section states in ADC-code units, 64-bit accumulators, and saturation after each section. Coefficients use round-half-away-from-zero. Section outputs truncate toward zero because round-to-nearest produced a persistent 127-code zero-input limit cycle in the feedback path. The model rechecks the quantized poles, detects accumulator or state overflow, tests zero-input decay, exports 24-bit hexadecimal filter vectors and per-channel threshold registers, and runs the detector with integer comparisons and counters. MAD threshold estimation remains an offline calibration operation. Outputs are written under `data/fixed_point/`.

Run the deterministic 100-trial randomized stress suite with:

```sh
.venv/bin/python scripts/stress_fixed_point.py --trials 100
```

The suite varies input shapes, amplitudes, channel counts, impulses, steps, extreme ADC codes, synthetic spikes, and detector event arrangements. It checks numerical error against the dequantized-coefficient float64 reference, arithmetic bounds, saturation, zero-input settling, repeatability, and exact detector behavior. The seed, per-trial measurements, explicit pass limits, and summary are written to `reports/fixed-point-stress.json`. Passing randomized tests increases confidence in the tested model but does not replace RTL equivalence or formal verification.

With seed `20261025`, all 100 trials passed. The worst RMS difference was 47.50 codes (0.196 µV), the worst absolute difference was 200.88 codes (0.828 µV), no section saturated, every zero-input tail settled exactly to zero, and the widest observed accumulator value required 47 bits including sign. The randomized detector checks exactly reproduced 611 merged events from 1,748 channel candidates. These results use deliberately harsh digital inputs, including full-scale steps and repeated extreme ADC codes; they are software-model validation, not neural detection accuracy measurements.

The first RTL exercise, `rtl/sample_input.sv`, is a one-entry ready/valid buffer for a signed ADC code and channel ID. It holds its output while `captured_ready` is low, accepts a replacement when the old sample is consumed, and clears its valid flag on consumption without replacement. Its self-checking testbench covers reset, stalls, signed limits, and simultaneous input/output handshakes. Run it with:

```sh
iverilog -g2012 -Wall -s sample_input_tb -o /tmp/neurosig-sample-input.vvp rtl/sample_input.sv rtl/tb/sample_input_tb.sv
vvp /tmp/neurosig-sample-input.vvp
```

The per-channel state storage for one biquad is in `rtl/biquad_state.sv`. It stores two prior signed 24-bit inputs and two prior outputs per channel, reads the selected channel combinationally, shifts one channel on a write, and clears all channels on reset. Run its self-checking testbench with:

```sh
iverilog -g2012 -Wall -s biquad_state_tb -o /tmp/neurosig-biquad-state.vvp rtl/biquad_state.sv rtl/tb/biquad_state_tb.sv
vvp /tmp/neurosig-biquad-state.vvp
```

The standalone Q2.30 biquad calculator is in `rtl/biquad_arithmetic.sv`. Its testbench compares against deterministic Python-generated integer vectors, including both actual filter coefficient sets, signed fractional values, saturation boundaries, and randomized samples. Reproduce its check with:

```sh
.venv/bin/python rtl/tb/generate_biquad_vectors.py --output /tmp/neurosig-biquad-vectors.txt
iverilog -g2012 -Wall -s biquad_arithmetic_tb -o /tmp/neurosig-biquad-arithmetic.vvp rtl/biquad_arithmetic.sv rtl/tb/biquad_arithmetic_tb.sv
vvp /tmp/neurosig-biquad-arithmetic.vvp +VECTORS=/tmp/neurosig-biquad-vectors.txt
```

The RTL source and testbenches are also copied into the user's separate `NeuroSignalProject` VS Code folder. Keep the copies synchronized until the workspace layout is consolidated. `rtl/biquad_controller.sv` joins one biquad's state and arithmetic with ready/valid handshakes; its testbench passes 49 checks. `rtl/spike_band_filter.sv` cascades independent 300 Hz high-pass and 6 kHz low-pass controllers at 32 kHz. Its integration smoke test passes for two channels, handshakes, and output stalls. The Python-to-RTL reference test compares 32,768 consecutive interleaved MEArec ADC samples (1,024 frames × 32 channels) with exact 24-bit results under patterned output stalls:

```sh
.venv/bin/python rtl/tb/generate_filter_vectors.py --output /tmp/neurosig-filter-vectors.txt
iverilog -g2012 -Wall -s spike_band_filter_reference_tb -o /tmp/neurosig-filter-reference.vvp rtl/spike_band_filter.sv rtl/biquad_controller.sv rtl/biquad_state.sv rtl/biquad_arithmetic.sv rtl/tb/spike_band_filter_reference_tb.sv
vvp /tmp/neurosig-filter-reference.vvp +VECTORS=/tmp/neurosig-filter-vectors.txt
```

All 32,768 comparisons passed on November 2, 2026. This verifies the tested recording segment and flow control pattern; a full-recording comparison remains future work.

The first one-channel detector module, `rtl/spike_detector_channel.sv`, has a directed testbench covering downward crossings, 16-sample peak search, tied minima, the final sample in the window, invalid clock gaps, reset, and a 32-sample refractory boundary. Its 77 sample/idle checks pass:

```sh
iverilog -g2012 -Wall -s spike_detector_channel_tb -o /tmp/neurosig-spike-detector-channel.vvp rtl/spike_detector_channel.sv rtl/tb/spike_detector_channel_tb.sv
vvp /tmp/neurosig-spike-detector-channel.vvp
```

The one-channel Python-to-RTL comparison uses the full 32,000-sample filtered trace and the matching offline threshold for each channel. Run the default channel 21 check with:

```sh
.venv/bin/python rtl/tb/generate_detector_vectors.py
iverilog -g2012 -Wall -s spike_detector_channel_reference_tb -o /tmp/neurosig-detector-reference.vvp rtl/spike_detector_channel.sv rtl/tb/spike_detector_channel_reference_tb.sv
vvp /tmp/neurosig-detector-reference.vvp +TRACE=/tmp/neurosig-detector-trace.txt +EVENTS=/tmp/neurosig-detector-events.txt
```

On November 2, 2026, the same check passed independently for all 32 channels (`--channel 0` through `--channel 31`): 1,024,000 sample comparisons and all 37 unmerged candidate events matched Python exactly, including event indices and signed codes. The test inserts periodic invalid clock cycles and uses Python-filtered input; the integrated test below covers the hardware filter path.

### Integrated RTL pipeline

`rtl/spike_detector_multichannel.sv` routes each accepted filtered sample to its channel detector and counts sample indices independently per channel. The threshold bus packs 32 signed 24-bit offline-calibrated codes, channel zero in the least significant slice. `rtl/spike_event_merge.sv` groups events within 16 sample frames of the earliest candidate, chooses the largest threshold margin, and waits for a sample-count watermark before releasing a group. Its 64-entry candidate queue exposes a sticky overflow output. `rtl/neurosig_pipeline.sv` connects the input buffer, two-stage filter, detector, and merger. `flush` is for end-of-stream draining and must be asserted after the final filtered sample has passed the detector.

The merge unit's directed test covers out-of-order candidate timestamps, anchored grouping, strength selection, flush, and concurrent append. Run it with:

```sh
iverilog -g2012 -Wall -s spike_event_merge_tb -o /tmp/neurosig-event-merge.vvp rtl/spike_event_merge.sv rtl/tb/spike_event_merge_tb.sv
vvp /tmp/neurosig-event-merge.vvp
```

To repeat the full end-to-end comparison on the one-second, 32-channel recording, generate the Python event vectors, build the timed Verilator simulation, and run it:

```sh
.venv/bin/python rtl/tb/generate_pipeline_reference.py
verilator --binary --timing -j 4 -Wno-fatal --top-module neurosig_pipeline_reference_tb --Mdir /tmp/neurosig-verilator-pipeline rtl/tb/neurosig_pipeline_reference_tb.sv rtl/neurosig_pipeline.sv rtl/sample_input.sv rtl/spike_band_filter.sv rtl/biquad_controller.sv rtl/biquad_state.sv rtl/biquad_arithmetic.sv rtl/spike_detector_multichannel.sv rtl/spike_detector_channel.sv rtl/spike_event_merge.sv
/tmp/neurosig-verilator-pipeline/Vneurosig_pipeline_reference_tb +ADC=data/quantized/adc_codes.hex +THRESHOLDS=/tmp/neurosig-thresholds.txt +EVENTS=/tmp/neurosig-merged-events.txt +OBSERVED=/tmp/neurosig-rtl-events.txt
.venv/bin/python rtl/tb/evaluate_pipeline_events.py
```

On November 2, 2026, Verilator 5.050 passed all 1,024,000 signed ADC codes and produced the same nine merged events as Python, matching each sample index, channel ID, and signed output code. Against MEArec ground truth with 1 ms one-to-one matching, the RTL produced 9 true positives, 0 false positives, and 0 false negatives; precision and recall were both 1.0 on this seeded recording. The measured result is in `reports/rtl-pipeline-validation.json`. RTL throughput, synthesized timing, area, and power have not been measured. The event queue depth and end-of-stream flush protocol still need evaluation under denser recordings and a system-level source. The channel detector needs 16 following valid samples to close a peak search; a crossing in the final 16 samples of a finite recording currently requires explicit padding or a separate end-of-stream search rule to match Python's shortened window.

MEArec reference: [installation documentation](https://mearec.readthedocs.io/en/latest/installation.html) and [source repository](https://github.com/SpikeInterface/MEArec).

## Project plan

1. **Validate the simulator.** Generate templates and a short recording with MEArec, retain generation parameters and random seeds, and inspect the recording alongside its spike ground truth.
2. **Build a 32-channel software reference.** Define sample rate, sample encoding, filtering, detection thresholds, and refractory behavior. Measure detection precision and recall against MEArec ground truth.
3. **Implement spike detection in SystemVerilog.** Export deterministic sample streams and expected results, then compare RTL output with the software reference. Document latency, throughput, and arithmetic limits.
4. **Choose a decoding task and train a small model.** Select labeled data, establish a software baseline, and evaluate fixed point inference before committing to an accelerator architecture.
5. **Implement the decoder in RTL.** Export weights and test vectors, verify numerical agreement, and measure resource use and performance.
6. **Evaluate the digital ASIC flow.** Synthesize the verified design and explore SKY130/OpenROAD timing, area, and power estimates. This is a future milestone; no physical design results exist yet.

MEArec supplies synthetic recordings and spike ground truth. It does **not** automatically supply intended movement labels for a motor decoder. Movement decoding will require a dataset with behavioral labels or a separately defined synthetic task whose limitations are documented.

## Working on the project

Keep simulation, software reference, RTL, and physical design milestones independently reproducible. Record commands, configuration, seeds, and measured results when completing a milestone. Update this README as the setup and validation status changes. See [AGENTS.md](AGENTS.md) for agent guidance and the current execution checklist.
