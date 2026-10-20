# Neurosig

Neurosig is a neural signal processing hardware project inspired by Neuropixels recordings. The plan is to generate synthetic extracellular recordings, implement spike detection in SystemVerilog, train a small neural decoder, and implement its inference in RTL. A later milestone is to evaluate the digital design using SKY130 and an open source physical design flow.

## Current status

The MEArec setup is working on Apple Silicon macOS with Python 3.12.15, MEArec 1.11.0, NEURON 8.2.7, and LFPy 2.3.6. The verified smoke test compiles NEURON mechanisms, generates two biophysical templates, and produces a one-second, 32-channel recording at 32 kHz with nine ground-truth spikes (seed 42). It reloads the HDF5 recording and checks its shape, finite nonconstant samples, and spike ground truth. See [the validation summary](reports/setup-validation.json).

The first recording target is **32 channels**. The smoke test uses one excitatory neuron and a Neuronexus-32 probe; larger datasets and the spike detection reference are the next milestone.

## Setup and validation

Run these commands from the repository root:

```sh
bash scripts/setup.sh
.venv/bin/python scripts/smoke_mearec.py
```

Setup requires network access and an existing Python 3 interpreter to bootstrap `uv`. Template generation requires a working C/C++ compiler and `make` (Xcode Command Line Tools on macOS). The current lock was verified on Apple Silicon macOS.

The setup uses a project-local `uv` executable under `.tools/` and a Python environment under `.venv/`. Dependency declarations live in `requirements.txt`; exact resolved versions are recorded in `requirements-lock.txt`. The smoke test writes generated artifacts under `data/smoke/`. These generated files and local environments should stay out of Git.

Outputs are `templates.h5`, `recording.h5` (including spike ground truth), `parameters.yaml`, and `summary.json`. The smoke script uses a temporary path alias for MEArec and NEURON subprocesses so this checkout can live in a folder containing spaces. It reads the packaged defaults directly and keeps generated models and compiler output temporary.

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
