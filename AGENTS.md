# Neurosig agent guidance

## Purpose and scope

Build a reproducible neural signal processing project: MEArec simulation → software reference → SystemVerilog spike detection → trained decoder → RTL inference → digital physical design exploration. Neuropixels is the inspiration; this repository does not currently implement a probe or analog acquisition hardware.

The initial recording target is 32 channels. Keep the channel count configurable and permit smaller smoke tests.

## Current execution checklist

- [x] Create a project-local Python environment and install MEArec with template generation dependencies.
- [x] Complete reproducible setup scripts and dependency locks.
- [x] Complete and verify a short MEArec template and recording generation smoke test.
- [x] Document the exact validation command and generated output summary.
- [x] Establish 32-channel recording parameters and a floating-point software spike detection reference.
- [ ] Interactive 3D visualization smoke: install `viz/requirements.txt`, build `viz/web`, run `.venv/bin/python -m viz.sim.server`, and run `viz.tests`.
- [ ] Define the sample stream interface, fixed point representation, and RTL verification vectors.
- [ ] Implement and verify the SystemVerilog spike detector.
- [ ] Select a labeled decoding task and establish a software baseline.
- [ ] Quantize and implement decoder inference in RTL.
- [ ] Explore synthesis and SKY130/OpenROAD implementation after functional verification.

Environment installed during initial setup: Python 3.12.15, MEArec 1.11.0, NEURON 8.2.7, and LFPy 2.3.6. On October 19, 2026, `.venv/bin/python scripts/smoke_mearec.py` passed on Apple Silicon macOS: two templates, 32,000 samples × 32 channels, one second at 32 kHz, and nine ground-truth spikes. `reports/setup-validation.json` records the result. Hardware results remain future work.

## Working rules

- Support the user's learning: explain concepts and give incremental instructions; let them choose ADC limits and calculate quantization before supplying a finished conversion implementation.
- `scripts/unfiltered_mearec.py` regenerates the smoke data with recording filtering disabled for ADC range inspection; outputs live under `data/unfiltered/`.
- The selected digital filter is a causal cascade of a second-order 300 Hz Butterworth high-pass and a second-order 6 kHz Butterworth low-pass at 32 kHz. `scripts/filter_iir.py` is the float64 reference. Recheck pole locations after every coefficient quantization change; float stability does not prove fixed-point stability.
- `scripts/detect_spikes.py` is the floating-point detector reference: per-channel MAD noise, negative six-sigma threshold, local-minimum timestamps, 1 ms refractory time, 0.5 ms cross-channel merge, and 1 ms ground-truth matching tolerance. Its 9/9 result is specific to the seeded one-second single-neuron smoke recording.
- `scripts/fixed_point_model.py` is the bit-accurate reference: Q2.30 coefficients, 24-bit signed section states at the ADC LSB scale, 64-bit accumulators, and section saturation. Coefficients round half away from zero; section outputs truncate toward zero because nearest rounding caused a 127-code zero-input limit cycle. The runtime detector uses integer threshold registers, comparisons, subtraction, and counters; MAD threshold generation is offline calibration.
- `scripts/stress_fixed_point.py` runs the reproducible randomized fixed-point stress suite. Preserve its seed, explicit error limits, per-trial report, zero-input decay check, and exact synthetic detector checks when changing arithmetic or event behavior.
- `rtl/sample_input.sv` and `rtl/biquad_state.sv` have standalone passing testbenches. The latter is one biquad's per-channel history, not the arithmetic filter. Matching copies live in the separate `NeuroSignalProject` VS Code folder; synchronize deliberately when editing either location.
- Read the README and inspect the current files before extending the project. Keep its status and this checklist consistent with actual results.
- Use `scripts/setup.sh` for environment setup and `.venv/bin/python` for project commands. Use the dependency files as the source of truth for package versions.
- Keep local tools, environments, generated datasets, compiled mechanisms, and large simulation outputs out of Git. Commit source, small configurations, and concise result summaries.
- Record random seeds, simulator parameters, versions, and commands for reproducibility. A smoke test demonstrates operation of its tested configuration only.
- Preserve MEArec spike ground truth for evaluating detection. Define the event matching tolerance and metrics before comparing detectors.
- MEArec does not supply intended movement labels. Select a labeled dataset or explicitly define a synthetic decoding task before reporting decoder accuracy.
- Define signedness, scaling, saturation or overflow behavior, ordering, handshake behavior, and reset behavior before implementing RTL interfaces. Compare RTL against a deterministic software reference.
- Report measured accuracy, latency, throughput, area, and power separately from targets or estimates. Document tool versions and constraints for physical design results.
- Keep the first implementation small enough to validate end to end. Expand channel count or model complexity after the existing pipeline is reproducible.
- Visualization code lives under `viz/` only. Do not edit golden-model arithmetic or RTL to support the UI. Neuropixels playback is sorted spikes, not ADC samples. Localization remains unimplemented until a validated algorithm exists.

## Parallel work

When tasks run in parallel, assign ownership of files, share interface assumptions early, and avoid modifying the same files concurrently. Integrate completed work with a validation command and a concise record of the outcome. Do not claim another task's pending check has passed.
