# Phase 5 floating-point detector, `detector-v2.0`

Config: `reports/phase5/detector-v2.json` (`configs.v2`, name `car_4sigma_200_4500_negative`).
Code: `scripts/phase5/detectors.py` (`P5Config`, `run_detector`). The Phase 4 frozen detector
(`car_4sigma_300_6000_negative`) is unchanged and is the default `P5Config()`; a test asserts the
two produce identical events.

**Differences from the frozen baseline: only the band-pass corner frequencies
(300–6000 Hz → 200–4500 Hz).** Every other stage is identical.

## Signal path (all float64, arrays are `[time, channel]`)

| Stage | Definition |
|---|---|
| Sample rate | 30 000 Hz (native IBL AP) |
| Input | ADC code × 2.34375 µV (native 10-bit probe ADC, gain 500). Voltages are exactly representable, so ADC quantisation adds no error on this data |
| Filter | Causal cascade: 2nd-order Butterworth high-pass 200 Hz → 2nd-order Butterworth low-pass 4500 Hz, from `scripts.filter_iir.design_sos` (same SOS design as the hardware filter), zero initial state |
| Common reference | Per sample, subtract the **median across channels** of the filtered values (applied *after* filtering). Even channel counts use the mean of the two middle values |
| Noise | Per channel on the CAR output: baseline = median, σ = MAD/0.6745, then 3 iterations excluding samples further than 6σ from the baseline and re-estimating |
| Threshold | `baseline − 4σ` (negative polarity only) |
| Crossing | Sample `t` with `x[t−1] ≥ thr` and `x[t] < thr` |
| Peak | Minimum of `x[t .. t+15]` (0.5 ms); first minimum on ties |
| Refractory | Per channel, 30 samples (1 ms): skip a crossing if `t_cross − last_peak < 30`, and drop a peak if `t_peak − last_peak < 30` |
| Artifact veto | Group candidates anchored on the first candidate within 6 samples (0.2 ms); drop the group if it spans ≥ max(2, ceil(0.5·N)) distinct channels |
| Merge | Group remaining events anchored on the first within 15 samples (0.5 ms); keep the largest `(baseline − x)/σ`; ties keep the earliest, then lowest channel |
| Output | `(sample, channel, voltage_uv, normalized_amplitude)` |

## Edge cases and reset

* The first sample can never be a crossing; a peak search at end-of-stream uses the samples available.
* Zero or non-positive σ on a channel is an error in the float reference (the integer model disables such a channel instead, see the fixed-point spec).
* State (filter, refractory) starts at zero at stream start. There is no mid-stream reset in the float reference.
* σ and baseline are estimated per window from the window itself (the Phase 4 convention). Calibrating on another window of the same session gave the same F1 (0.553 both ways, `reports/phase5/calibration_transfer_heldout.json`).

## Limitations

* Validated on 32 contiguous visual-cortex contacts per insertion; four public insertions.
* Agreement is measured against a spike sorter that omits low-amplitude and multi-unit activity. Nothing here is biological ground truth.
