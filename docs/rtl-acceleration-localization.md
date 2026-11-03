# Localization RTL acceleration (proposal only)

Python localization is the verified baseline. **Do not implement localization RTL until this baseline is independently bit-accurate and a hardware interface freeze exists.** Existing `rtl/neurosig_pipeline.sv` and `rtl/spike_count_decoder.sv` stay the detection and count-decoder comparison points.

Measured on seed `20261103` by `scripts/run_localization_benchmark.py` (576 estimates, 32 channels, 32 kHz). Times are median wall-clock on this host, not ASIC latency.

| Method | Isolated low-noise median error | Sweep median error (noise + overlap) | Median latency | Approx. MACs / event |
|---|---:|---:|---:|---:|
| amplitude barycenter + 1/r fit | 6.75 µm | 63.6 µm | 1.03 ms isolated / 0.98 ms sweep | ~27k (`49×17` radial grid × 32 channels) |
| template cosine match | 12.9 µm | 42.4 µm | 0.071 ms / 0.063 ms | ~8k (`240` templates × 32) |
| quasi-static amplitude scan (`beamform`) | 6.51 µm | 54.0 µm | 0.196 ms / 0.188 ms | ~308k (`17×63×9` grid × 32) |

The scan is **not** RF delay-and-sum. Steering vectors are predicted monopole **amplitudes**. Extracellular delays are unused.

## Memory and streaming

- Snippet: 1.0 ms pre + 1.5 ms post → 80 samples × 32 channels × 16-bit ≈ 5 KiB per event, sample-major, matching NSG1 `SNIPPET` packets.
- Amplitude vector is 32 float/int peaks; the 3D grid and template catalog dominate storage (~9.6k × 32 floats for the default scan).
- Event packets are 16 B header + 20 B payload + 4 B CRC.

## Stay on the edge PC

- Template catalog construction and held-out matching policy.
- 3D grid search / beamform-inspired scan and any covariance/uncertainty.
- Decoder **training**, spatial feature assembly, visualization, and reports.
- Coordinate transforms into stylized display millimetres.

## Reasonable later ASIC candidates (not scheduled)

1. **NSG1 packetizer** after `rtl/spike_event_merge.sv`: EVENT mode A, optional SNIPPET DMA of a circular per-channel buffer. Reuse `rtl/sample_input.sv` ready/valid.
2. **Peak / RMS extraction** on the filtered 24-bit stream already in the pipeline.
3. **Integer barycenter** of thresholded channel amplitudes (shifts/adds; Q format TBD after a bit-accurate Python model).
4. **Streaming matched-filter MAC** against a small on-chip amplitude template set (far smaller than the 9.6k-point scan).

Fixed-point requirements are **not** frozen. A later model must specify signedness, peak scaling (ADC LSB vs µV), saturation, and compare against `edge/` exactly. Nearest rounding is not assumed; the IIR path already showed limit-cycle sensitivity to rounding.

`rtl/spike_count_decoder.sv` remains the 32-count baseline. Spatial features (43-d in the Python experiment) are edge-only until a separate feature map and scores match a software reference.

## What this proposal does not claim

- No area, power, or clock numbers (none were synthesized).
- No RTL equivalence for localization or the NSG1 codec.
- Localization error on the toy monopole is not biophysical accuracy and is not a stimulation targeting specification.
