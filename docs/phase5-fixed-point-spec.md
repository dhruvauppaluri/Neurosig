# Phase 5 fixed-point specification, `fixed-v1`

Reference: `scripts/phase5/fixed_detector.py`. Two independent implementations (vectorised
`run_fixed` and scalar frame-by-frame `StreamingFixed`) must agree bit for bit; `tests/test_phase5_fixed.py`
checks this, and `tests/golden/phase5_fixed_golden.json` pins outputs (those hashes pin current behaviour;
they are not an independent oracle). **No RTL implements this model; nothing here claims RTL verification.**

All values are two's-complement integers. `>>` is an arithmetic shift (floor). `rsr(v,b) = (v + 2^(b-1)) >> b`
(round half up). Python integers are unbounded; the widths below were checked against measured maxima.

## Data formats

| Signal | Type | Scale / rule |
|---|---|---|
| ADC code | s16 container (native range ±512) | `round_half_away(v_uV / 2.34375)`, saturate to [−32768, 32767] |
| Internal signal | s24 | `adc << 10` (10 guard bits). LSB = 2.34375/1024 = 0.00228882 µV; range ±8192 ADC codes |
| IIR coefficients | s32 Q2.30 | `round_half_away(coef·2^30)`, `a0 = 2^30` (reuses `fixed_point_model.quantize_sos`) |
| IIR state, section output | s24 | `trunc_toward_zero(acc >> 30)` then saturate to s24 (unchanged semantics from `fixed_point_model`) |
| IIR accumulator | s64 | `b0·x0 + b1·x1 + b2·x2 − a1·y1 − a2·y2`. Measured worst case 50 bits on real data; alternating full-scale input fits in ≤ 64 |
| CAR output | s24 | `filtered − ref`, saturate to s24 |
| Baseline register | s24 | integer median of the CAR output (see Calibration) |
| MAD register | u24 | integer MAD |
| σ register | u41 | `mad · 97157` = σ in units of 2^−16 internal LSB (`97157 = round(2^16/0.6744897502)`) |
| Threshold register | s24 | `baseline − rsr(thr_mult_q8 · σq16, 24)`; `thr_mult_q8 = 1024` for 4.0σ; clipped to [−2^23+1, 2^23−1] |
| Reciprocal register | u32 | `floor(2^48 / σq16)`, 0 for disabled channels |
| Event timestamp | u32 | frame counter; wraps at 2^32 (39.8 h at 30 kHz) — wrap is not handled by the reference |
| Event amplitude | s25 `delta` | `baseline − code`, always > 0 |
| Event normalised amplitude | u16 Q8.8 σ | `min(0xFFFF, (delta · recip) >> 24)` |
| Channel id | u8 | 0 … 255 |

## Common reference (`car`)

* **median** (exact): sort the active channels' s24 outputs for the frame. Odd N: middle element. Even N: `(lo + hi) >> 1`
  (floor of the mean of the two middle values; e.g. `(−3 + 0) >> 1 = −2`).
* **mean** (separate variant): `ref = (Σ x · R + 2^31) >> 32`, `R = 2^32/N` if N is a power of two else `round(2^32/N)`.
* **none**: `ref = 0`.
* Disabled channels are excluded from the reference but are still filtered; their threshold is the sentinel `−2^23`
  (`x < −2^23` is impossible) so they never fire.

## Offline calibration (integer, deterministic)

Per channel on a calibration segment of CAR outputs `x`:
1. `b = int_median(x)`, `m = int_median(|x − b|)` where an even-length median is `(lo + hi) >> 1`.
2. 3 times: `lim = (m · 582964) >> 16` (6σ in MAD units); keep `|x − b| ≤ lim` (fall back to all samples if fewer than 32 remain); recompute `b`, `m` on the kept samples.
3. `σq16 = m · 97157`; `thr = b − rsr(1024 · σq16, 24)`; `recip = 2^48 // σq16`.
4. A channel with `m = 0` is automatically disabled (`thr = −2^23`, `recip = 0`).

## Runtime detector (per active channel, per valid frame)

* Crossing: previous valid CAR value `≥ thr` and current `< thr`; none on the first sample.
* Peak: minimum over frames with timestamp in `[t_cross, t_cross + 15]`; first minimum on ties.
* Refractory: ignore a crossing if `t_cross − last_peak < 30`; drop a peak if `t_peak − last_peak < 30`. `last_peak` initialises to `−31`.
* Veto and merge: as in the float spec with windows 6 and 15 samples; veto group needs `max(2, (128·N_active + 255) >> 8)` distinct channels; merge keeps the largest `amp_q8` (earliest, then lowest channel on ties).

## Disabled channels, dropped frames, reset

* **Dropped frame** (`valid = 0`): filter state holds, no comparison, the timestamp counter still advances, open peak-search windows still expire by timestamp, and "previous value" for crossing detection is the last valid frame.
* **Reset**: all filter states, `last_peak = −31`, previous-value flags and open windows clear. Calibration registers are inputs and are not reset.
* **End of stream / flush**: every open peak search closes with the minimum seen so far.

## Why 10 guard bits (independent validation of a changed convention)

The existing project filter works at the ADC LSB scale. On 2.34 µV real data that scale is too coarse: truncation error
enters inside the high-pass section's feedback loop where the noise gain is of order 10³ LSB. Measured on three calibration
windows (float vs integer, same ADC codes):

| Guard bits | Filter error RMS (µV) | Candidates identical to float | Events identical to float |
|---|---|---|---|
| 4 | 3.0 – 6.0 | 2152 / 2910 | 1033 / 1491 |
| 6 | 0.9 – 1.7 | 2635 / 2910 | 1325 / 1491 |
| 8 | 0.25 – 0.43 | 2827 / 2910 | 1437 / 1491 |
| 10 | 0.06 – 0.11 | 2885 / 2910 | 1477 / 1491 |

`signal_bits = 24`, Q2.30 coefficients, 64-bit accumulators and truncation-toward-zero are unchanged. Zero-input decay to exactly 0 and
quantised-pole stability are tested for both 300/6000 Hz and 200/4500 Hz (`max |pole| = 0.9708` for v2).

## Quantised v2 coefficients (Q2.30, rows `b0 b1 b2 a0 a1 a2`)

```
HP 200 Hz : 1042404711 -2084809422 1042404711 1073741824 -2083894655 1011982365
LP 4500 Hz:  140774468   281548936  140774468 1073741824  -802932516  292288564
```

## Deterministic-behaviour statement

Given the same ADC frames, valid mask, active mask, spec and calibration registers, both implementations produce identical
events. There is no floating point, randomness, or dictionary/ordering dependence in the integer path. What is *not* specified:
timestamp wrap, a hardware event-queue depth, and cycle-level timing.
