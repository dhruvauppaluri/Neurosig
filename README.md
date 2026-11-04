# Neurosig

Neurosig is a neural signal processing hardware project inspired by Neuropixels recordings. The plan is to generate synthetic extracellular recordings, implement spike detection in SystemVerilog, train a small neural decoder, and implement its inference in RTL. A later milestone is to evaluate the digital design using SKY130 and an open source physical design flow.

## Current status

The MEArec setup is working on Apple Silicon macOS with Python 3.12.15, MEArec 1.11.0, NEURON 8.2.7, and LFPy 2.3.6. The verified smoke test compiles NEURON mechanisms, generates two biophysical templates, and produces a one-second, 32-channel recording at 32 kHz with nine ground-truth spikes (seed 42). It reloads the HDF5 recording and checks its shape, finite nonconstant samples, and spike ground truth. See [the validation summary](reports/setup-validation.json).

The first recording target is **32 channels**. The smoke test uses one excitatory neuron and a Neuronexus-32 probe. The floating-point and fixed-point software references and a complete RTL filter and spike detector pipeline are implemented. The seeded one-second end-to-end RTL test matches all nine Python merged events exactly. A 32-feature linear decoder has been trained and verified in RTL on both synthetic spike-count windows and a small processed real Neuropixels session. SVA k-induction in [`formal/`](formal/README.md) proves the decoder score and the biquad controller handshake.

A separate research UI under [`viz/`](viz/README.md) shows a translucent anatomical human brain surface, illustrative synthetic activity and wiring, and fixture Neuropixels **sorted-spike** playback confined to the visual region. It compares five localization methods on the same event and shows offline benchmark provenance. A Python `edge/` package localizes synthetic, MEArec, and IBL sorter-triggered waveforms; emulates an NSG1 ASIC-to-edge link; and compares count, geometry, estimated-location, and tiny-MLP decoding. Localization RTL is **not** implemented. The [Phase 2](reports/phase2-validation.md) and [Phase 3](reports/phase3-validation.md) validation records separate measured results from limits.

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

On November 2, 2026, Verilator 5.050 passed all 1,024,000 signed ADC codes and produced the same nine merged events as Python, matching each sample index, channel ID, and signed output code. Against MEArec ground truth with 1 ms one-to-one matching, the RTL produced 9 true positives, 0 false positives, and 0 false negatives; precision and recall were both 1.0 on this seeded recording. The measured result is in `reports/rtl-pipeline-validation.json`. Synthesized timing, area, and power have not been measured. The November 3 stress and finite-recording tests are described below.

### Cycle throughput

`rtl/tb/neurosig_throughput_tb.sv` drives the complete pipeline with `sample_valid` held high and advances the channel and sample only after a rising-edge ready/valid handshake. It uses 4,096 frames × 32 channels of zero ADC codes, so the test measures the steady input acceptance rate without file I/O or event-queue activity. Reproduce it with:

```sh
verilator --binary --timing -j 4 -Wno-fatal --top-module neurosig_throughput_tb --Mdir /tmp/neurosig-verilator-throughput rtl/tb/neurosig_throughput_tb.sv rtl/neurosig_pipeline.sv rtl/sample_input.sv rtl/spike_band_filter.sv rtl/biquad_controller.sv rtl/biquad_state.sv rtl/biquad_arithmetic.sv rtl/spike_detector_multichannel.sv rtl/spike_detector_channel.sv rtl/spike_event_merge.sv
/tmp/neurosig-verilator-throughput/Vneurosig_throughput_tb +REPORT=reports/rtl-throughput.json
```

On November 2, 2026, Verilator 5.050 accepted 131,072 samples in 393,212 inclusive cycles: 2.999969 cycles/sample on average, with a maximum 3-cycle gap between accepted samples. At an **assumed** 100 MHz clock, this is 33.334 million samples/s, above the 1.024 million samples/s target. Equivalently, the observed cycle rate would need an approximately 3.072 MHz clock to meet the average target. These are cycle-simulation results and a clock-rate projection; synthesis and static timing analysis have not shown that the design meets any specific clock frequency. The measurements are in `reports/rtl-throughput.json`.

MEArec reference: [installation documentation](https://mearec.readthedocs.io/en/latest/installation.html) and [source repository](https://github.com/SpikeInterface/MEArec).

### Decoder baseline and RTL inference

`scripts/synthetic_decoder.py` defines a reproducible **synthetic** binary latent-state task from 100 ms windows of 32 Poisson spike-count channels. With seed `20261102`, a held-out 1,024-example test achieved 94.5% float and fixed accuracy, with all predictions agreeing. The weights are quantized to signed 16-bit Q8.8 and the bias to signed 32-bit Q8.8. `rtl/spike_count_decoder.sv` sums 32 unsigned 8-bit count × signed weight products with the bias and classifies by score sign. All 1,024 RTL scores matched Python exactly. This task is a hardware exercise and does not establish movement-decoding performance.

```sh
.venv/bin/python scripts/synthetic_decoder.py
iverilog -g2012 -Wall -s spike_count_decoder_tb -o /tmp/neurosig-decoder.vvp rtl/spike_count_decoder.sv rtl/tb/spike_count_decoder_tb.sv
vvp /tmp/neurosig-decoder.vvp +WEIGHTS=data/decoder/weights.txt +VECTORS=data/decoder/rtl_vectors.txt
```

For a test with **real processed Neuropixels data**, download the small single-session extract from [Steinmetz et al. 2019 via Programming in Neuroscience](https://programminginneuroscience.com/2026/09/30/does-this-neuron-care-about-the-stimulus/):

```sh
mkdir -p data/neuropixels
curl -L --fail https://programminginneuroscience.com/wp-content/uploads/2026/09/steinmetz_session11.zip -o data/neuropixels/steinmetz_session11.zip
.venv/bin/python scripts/neuropixels_decoder.py
iverilog -g2012 -Wall -s spike_count_decoder_tb -o /tmp/neurosig-decoder.vvp rtl/spike_count_decoder.sv rtl/tb/spike_count_decoder_tb.sv
vvp /tmp/neurosig-decoder.vvp +WEIGHTS=data/neuropixels/decoder/weights.txt +VECTORS=data/neuropixels/decoder/rtl_vectors.txt +EXPECTED=73
vvp /tmp/neurosig-decoder.vvp +WEIGHTS=data/neuropixels/decoder/weights.txt +VECTORS=data/neuropixels/decoder/all_rtl_vectors.txt +EXPECTED=240
```

The downloaded file's SHA-256 was `9c055f1248670cd39c15f6ebba5e1d5dac72cda8a15778221aaf8fc379d6417b`. The task is blank versus full right-side visual contrast, using counts from the first 32 VISp units in a 50–250 ms post-stimulus window. A seed `20261103` stratified split gave 167 train and 73 held-out trials. Float and fixed models both scored 90.4% on the held-out trials versus a 69.9% majority-class baseline; all 73 held-out RTL scores and all 240 eligible-trial RTL scores matched fixed Python exactly. See `reports/neuropixels-decoder.json`. These are sorted, trial-binned spikes, **not raw voltage**: they stress the decoder but cannot validate the ADC, filter, or spike detector. This is visual-stimulus decoding in one mouse session, not intended movement decoding.

### Formal verification

SVA contracts for the two RTL blocks live in [`formal/`](formal/README.md). `formal/spike_count_decoder_formal.sv` checks that the 32-channel Q8.8 score matches a 64-bit dot product wrapped to 32 bits, and that `prediction` is the non-negative class, including a zero score. `formal/biquad_controller_formal.sv` checks the IDLE → COMPUTE → OUTPUT handshake, two-cycle latency, backpressure, reset, illegal-state recovery, and per-channel delay-line isolation. The controller proof replaces `biquad_arithmetic` with an unconstrained `new_y`, so the Q2.30 multiply-accumulate stays with `rtl/tb/biquad_arithmetic_tb.sv`.

```sh
OSS_CAD_SUITE=/path/to/oss-cad-suite bash formal/run_formal.sh
```

On November 4, 2026, SymbiYosys 0.70 with Yosys 0.70+329 (`b8d84f823`), the slang frontend, and Bitwuzla 0.9.1 proved both tasks by k-induction and reached every cover. The record is `reports/formal-verification.json`. This is a proof of those contracts. It is not a timing, area, or power result, and it does not prove the full filter-plus-detector pipeline.

### Edge integration, localization, and raw-data pilot

Python localization, a software NSG1 packet link, and an edge decoder comparison live under [`edge/`](edge/). They do not modify verified DSP arithmetic or RTL. Interfaces are listed in [`docs/edge-interfaces.md`](docs/edge-interfaces.md). A later hardware sketch is [`docs/rtl-acceleration-localization.md`](docs/rtl-acceleration-localization.md).

The synthetic forward model is a documented quasi-static monopole with a triphasic kernel (`edge/forward.py`). The newer benchmark also uses the seeded MEArec biophysical recording. Errors are Euclidean micrometers. Viewer millimetres use `edge.coords`.

```sh
.venv/bin/python -m unittest discover -s edge/tests
.venv/bin/python scripts/run_localization_benchmark.py
.venv/bin/python scripts/run_edge_decode.py
.venv/bin/python -m edge.benchmark_v2
.venv/bin/python -m edge.benchmark_v3
.venv/bin/python -m edge.decode.experiment_v2
.venv/bin/python scripts/run_edge_integration.py
./scripts/setup_ibl.sh
data/ibl-venv/bin/python -m edge.ibl
data/ibl-venv/bin/python -m edge.ibl_eval
data/ibl-venv/bin/python -m edge.decode.ibl_experiment
.venv/bin/python scripts/freeze_real_detector.py
data/ibl-venv/bin/python -m edge.decode.detector_experiment
```

The Phase 3/4 multi-seed benchmark uses 324 shared held-out events across linear and staggered probes, source distance, noise, and overlap. After aligned-trough amplitudes and interpolating scan peaks, median/p90 errors were 16.6/51.4 µm (amplitude), 18.7/41.5 µm (template), 16.4/43.2 µm (interpolated spatial scan), and 16.8/52.2 µm (weighted scan). Robust scan now **fails closed** on bimodal trough footprints (42/324 unavailable); on the remaining events median/p90/worst were 16.3/47.6/101.7 µm. Template matching remains the recommended edge algorithm. Historical v2 numbers stay in `reports/localization-benchmark-v2.json`. The seeded MEArec nine-spike check ran in this environment: 9/9 detections, template median 29.5 µm.

The software link encodes little-endian `NSG1` EVENT and SNIPPET frames with CRC32, a 64-deep drop-oldest FIFO, sequence-gap detection, and a control path for thresholds, channel masks, acquisition mode, status, and ping. `EventAssembler` joins EVENT and SNIPPET in either order by session-scoped event ID and emits once at end of stream. It bounds event state, reports missing partners, and rejects conflicting duplicates. `scripts/run_edge_integration.py` compared nine actual RTL event triples with Python, then used a **software NSG1 packet emulator** and MEArec snippets for edge localization and decoder scoring; the RTL does not output NSG1 frames. See `reports/edge-integration.json`.

The no-leakage synthetic A/B/C/D task uses counts, counts plus electrode geometry, counts plus **estimated** positions, and a 16-unit tanh MLP on spatial features (seeds 20261103–05). Mean held-out accuracy was 81.38%, 81.38%, 81.64%, and 81.25%. Ground-truth positions are not decoder inputs. The old `reports/edge-decode.json` is retained as a historical single-seed result; use `reports/edge-decode-v2.json` for the comparison. The Q8.8 RTL count decoder is a separately labeled baseline.

The IBL pilot uses public insertion `4836a465-c691-4852-a0b1-dcd2b1ce38a1` (`caa5dddc-9290-4e27-9f5e-575ba3598614`, probe00), native 30 kHz AP voltage, 32 VISp contacts, sorting revision `2024-05-06`, and trial revision `2025-03-03`. A second public insertion `1a60a6e1-da99-4d4e-a734-39b1d4544fad` (`51e53aff-1d5d-4182-a684-aba783d50ae5`, NYU-45, VISa) is used for cross-recording checks; channel selection prefers VISp and falls back to VIS*. Sorter peak-channel depth is an imperfect position proxy. The 358-trial left-versus-right visual-stimulus decoder uses sorted counts and processed waveform-derived location features with occupancy bins on the native contact y range (~2656–3014 µm). Across three splits, A/B/C/D mean accuracy was 70.37%/70.68%/70.99%/72.22% versus 57.41% majority baseline. McNemar tests of C or D versus counts were not significant (p=1.0; few discordant trials). These trials are distinct from the raw AP chunks, so this is not a raw-voltage-to-decoder accuracy result. A separate frozen-detector experiment bins 50–250 ms contact events from 40 s AP blocks: 11 eligible trials on the first insertion (held-out sorted 60% vs detector 20% on five trials; McNemar p=0.625) and 10 trials on NYU-45 (train-primary/test-secondary detector 70% equals the 70% majority; sorted 30%). That n is too small to rank detectors. See `reports/ibl-pilot.json`, `reports/ibl-detection-localization.json`, `reports/ibl-decode.json`, `reports/frozen-real-detector.json`, and `reports/ibl-detector-decode.json`. Large IBL downloads remain ignored under `data/`.

On the original t=100–101 s chunk the unchanged 6σ no-CAR detector still found one event and matched none of 294 sorting spikes within 1 ms. After median CAR, sorter-aligned troughs have median 3.42σ and only 7.5% exceed 6σ, so the MEArec threshold is the primary failure mode. The frozen Python detector is **median CAR + 300–6000 Hz + 4σ negative** (`car_4sigma_300_6000_negative`). It found 236 events and matched 110/294 sorter spikes at 1 ms (precision 0.47, recall 0.37) on that window. Across three disjoint 1 s windows on the first insertion, mean 1 ms recall versus sorting was 0.034 (legacy) versus 0.490 (frozen). On two 1 s windows of the second insertion the same freeze scored 0.081 versus 0.469. Sorting is an incomplete reference. **RTL does not follow**: median CAR is not in `rtl/neurosig_pipeline.sv`. The raw waveform localization comparisons use sorter-triggered snippets.

### Integrated RTL stress and finite-recording behavior

`rtl/tb/generate_pipeline_stress.py` makes seeded dense ADC streams and exact Python event vectors. With seed `20261103`, 256 frames at 1, 4, and 32 channels all passed end-to-end RTL comparison: respectively 5, 20, and 160 channel candidates merged to 5 events in each case. The detector's `end_of_stream` closes a partial 16-sample peak search at the end of a finite recording; the pipeline sends this signal to one channel per clock before flushing the event merge. Stop input and let the filter drain before asserting `flush`, and reset between recordings. Directed tests cover partial searches, a midstream integrated reset, and sticky queue overflow at 65 candidates with queue depth 64. See `reports/rtl-stress.json`. A system-level source still needs to exercise queue sizing under varied event densities.

For a dense case, run `.venv/bin/python rtl/tb/generate_pipeline_stress.py --channels 4 --frames 256 --prefix /tmp/ns-stress4`, compile `rtl/tb/neurosig_pipeline_reference_tb.sv` with Verilator `-GCHANNELS=4 -GFRAMES=256` and the same source list as the integrated test above, then pass `+ADC=/tmp/ns-stress4-adc.txt +THRESHOLDS=/tmp/ns-stress4-thresholds.txt +EVENTS=/tmp/ns-stress4-events.txt +OBSERVED=/tmp/ns-stress4-observed.txt`. Repeat with 1 and 32 channels. The directed checks are `rtl/tb/spike_detector_channel_tb.sv`, `rtl/tb/spike_event_merge_tb.sv`, and `rtl/tb/neurosig_reset_tb.sv`.

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
