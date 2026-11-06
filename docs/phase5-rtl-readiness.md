# Phase 5 RTL readiness and interface specification

Scope: what a *separate* RTL phase would have to change to reproduce `fixed-v1`
(`docs/phase5-fixed-point-spec.md`) exactly. **No RTL, FPGA or synthesis file was read for modification or changed in this phase;
every cost below is an analytical estimate, not a synthesis result, and nothing here is RTL-verified.**

## Gap analysis against the current pipeline (`rtl/neurosig_pipeline.sv`)

| Item | Current RTL | `fixed-v1` needs |
|---|---|---|
| Input | one `(channel_id, s16 code)` per transfer, 3 cycles/sample measured | same; `<< 10` guard-bit shift into the s24 filter input (the filter input is already a sign-extended 24-bit word; ±512 × 1024 = 2^19 fits) |
| Filter coefficients | hard-coded 300/6000 Hz at 32 kHz (`spike_band_filter.sv`) | 200/4500 Hz at **30 kHz** (values in the spec); the 300/6000 Hz 30 kHz set also differs from the 32 kHz constants. Make them parameters or load registers |
| Common reference | none | new frame-synchronous CAR stage after the filter |
| Threshold | `threshold_codes[CHANNELS*24]` input, compare `x < thr` | same register file, but loaded from the integer calibration (needs CAR first); add per-channel `armed` bit and reciprocal register |
| Peak search / refractory | counters at 32 kHz window sizes | 15 / 30 samples at 30 kHz (parameters) |
| Event merge key | `candidate_margin = thr − code` (not σ-normalised) | `amp_q8 = (delta · recip) >> 24`, `delta = baseline − code`; ties → earliest timestamp then lowest channel (current RTL breaks ties by earliest index only) |
| Artifact veto | none | grouped-in-0.2 ms, ≥ `max(2, ceil(N/2))` distinct channels |
| Dropped frames / enables | none | `valid` per frame, `active` mask per channel |

## Proposed module breakdown

1. `frame_assembler` — collects the 32 filtered s24 outputs of one time index (the filter emits channels serially). 32 × 24 b = 768 b double buffer.
2. `car_median` (or `car_mean`) — consumes a complete frame, outputs `ref` (s24). Exact median per the spec.
3. `car_subtract_sat` — `x − ref`, saturate s24, streams channels to the existing per-channel detector.
4. `detector_channel` (existing structure) — add: armed gate, per-channel `baseline`/`recip` registers, `delta`/`amp_q8` outputs, drop-frame behaviour.
5. `artifact_veto` — 6-sample window, distinct-channel counter on a 32-bit channel bitmap.
6. `event_merge` (existing) — change key to `amp_q8`, tie rule as above.
7. `calib_regs` — memory-mapped per-channel `{thr s24, baseline s24, recip u32, armed}`; calibration itself stays offline (software), as in the existing project.

### Interface sketch (new stage)

```
car_stage (CHANNELS = 32)
  in : clk, rst_n, in_valid, in_ready, in_channel[5], in_sample s24, in_frame_start
       cfg_active[32], cfg_mode[1:0] {0 none, 1 median, 2 mean}
  out: out_valid, out_ready, out_channel[5], out_sample s24   (x - ref, saturated)
       out_frame_ts u32
```

## Buffering, latency and throughput

* **Buffering:** CAR is instantaneous across channels, so no time history is needed, only one frame (768 b) plus the in-flight frame. The detector needs the previous CAR value per channel (already present as a per-channel state).
* **Latency:** one frame period (33.3 µs at 30 kHz) for frame assembly + median, plus the existing peak-search latency (≤ 16 samples = 0.53 ms) and veto/merge hold-off (≤ 21 more samples). Event timestamps are the trough sample and are unchanged by these delays.
* **Throughput at the measured 3 cycles/sample:** 32 channels × 3 = 96 cycles per frame against 3333 cycles available per 30 kHz frame at an *assumed* 100 MHz clock, so a serial median fits with a >30× margin.
* **Median cost (estimate):** a pipelined 32-input Batcher odd-even sorting network is 191 compare-exchanges of 24-bit values per frame; a serial rank-selection (24 bit-slices × 32 comparisons ≈ 768 cycles) uses almost no area. Mean CAR is 31 adders plus a constant multiply (a shift for N = 32). Neither needs RAM.
* **Mean-CAR trade-off (measured, 20 held-out windows, v2):** channel-consistent F1 0.539 vs 0.553, recall 0.438 vs 0.490, precision 0.700 vs 0.636, false detections/s 72 vs 108. F1 difference to the baseline is within noise; the detection behaviour differs (fewer detections). It is a different operating point, not a free simplification, and is kept as a separate variant.

## Overflow and state risks

* The IIR accumulator measured ≤ 50 bits on real data; the 64-bit accumulator holds worst-case alternating full-scale input (tested). Section outputs saturate at ±8192 ADC codes (16× the 10-bit range) and no saturation occurred in 20 held-out windows.
* Truncation-toward-zero is required: round-to-nearest produced a limit cycle in the earlier project. Zero-input decay to exactly 0 is tested for both coefficient sets.
* 10 guard bits are required for agreement with float (see spec table); 4 guard bits gave 3–6 µV RMS filter error.
* Timestamp wraps at 2^32 samples (39.8 h); `merge`/`veto` windows assume monotone timestamps.
* `recip = 2^48 // σq16` fits u32 whenever MAD ≥ 1 internal LSB (σq16 ≥ 97157 gives recip ≤ 2.9·10^9 < 2^32); a channel with MAD = 0 is disarmed by calibration and never divided.

## Verification plan for the later RTL phase

1. Replay `tests/golden/phase5_fixed_golden.json` and real-window ADC code vectors through the RTL; compare filter output, CAR output, thresholds, candidates and merged events exactly.
2. Use the corner-case list in `tests/test_phase5_fixed.py` (equality boundaries, refractory ±1, dropped frames, disabled channels, saturation) as directed tests.
3. Keep Python-first: change the spec only with a golden-vector update and a float-vs-fixed rerun (`compare_float_fixed.py`).
