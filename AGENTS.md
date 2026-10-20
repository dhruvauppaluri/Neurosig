# Neurosig agent guidance

## Purpose and scope

Build a reproducible neural signal processing project: MEArec simulation → software reference → SystemVerilog spike detection → trained decoder → RTL inference → digital physical design exploration. Neuropixels is the inspiration; this repository does not currently implement a probe or analog acquisition hardware.

The initial recording target is 32 channels. Keep the channel count configurable and permit smaller smoke tests.

## Current execution checklist

- [x] Create a project-local Python environment and install MEArec with template generation dependencies.
- [x] Complete reproducible setup scripts and dependency locks.
- [x] Complete and verify a short MEArec template and recording generation smoke test.
- [x] Document the exact validation command and generated output summary.
- [ ] Establish 32-channel recording parameters and a software spike detection reference.
- [ ] Define the sample stream interface, fixed point representation, and RTL verification vectors.
- [ ] Implement and verify the SystemVerilog spike detector.
- [ ] Select a labeled decoding task and establish a software baseline.
- [ ] Quantize and implement decoder inference in RTL.
- [ ] Explore synthesis and SKY130/OpenROAD implementation after functional verification.

Environment installed during initial setup: Python 3.12.15, MEArec 1.11.0, NEURON 8.2.7, and LFPy 2.3.6. On October 19, 2026, `.venv/bin/python scripts/smoke_mearec.py` passed on Apple Silicon macOS: two templates, 32,000 samples × 32 channels, one second at 32 kHz, and nine ground-truth spikes. `reports/setup-validation.json` records the result. Hardware results remain future work.

## Working rules

- Read the README and inspect the current files before extending the project. Keep its status and this checklist consistent with actual results.
- Use `scripts/setup.sh` for environment setup and `.venv/bin/python` for project commands. Use the dependency files as the source of truth for package versions.
- Keep local tools, environments, generated datasets, compiled mechanisms, and large simulation outputs out of Git. Commit source, small configurations, and concise result summaries.
- Record random seeds, simulator parameters, versions, and commands for reproducibility. A smoke test demonstrates operation of its tested configuration only.
- Preserve MEArec spike ground truth for evaluating detection. Define the event matching tolerance and metrics before comparing detectors.
- MEArec does not supply intended movement labels. Select a labeled dataset or explicitly define a synthetic decoding task before reporting decoder accuracy.
- Define signedness, scaling, saturation or overflow behavior, ordering, handshake behavior, and reset behavior before implementing RTL interfaces. Compare RTL against a deterministic software reference.
- Report measured accuracy, latency, throughput, area, and power separately from targets or estimates. Document tool versions and constraints for physical design results.
- Keep the first implementation small enough to validate end to end. Expand channel count or model complexity after the existing pipeline is reproducible.

## Parallel work

When tasks run in parallel, assign ownership of files, share interface assumptions early, and avoid modifying the same files concurrently. Integrate completed work with a validation command and a concise record of the outcome. Do not claim another task's pending check has passed.
