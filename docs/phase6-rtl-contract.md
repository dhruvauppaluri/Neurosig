# NeuroSig frozen RTL reference contract: `fixed-v1.1` / `detector-v2.0`

Status: **frozen software reference** for a future SystemVerilog implementation. **No RTL implements this model, and nothing here claims RTL verification.**
Supersedes `docs/phase5-fixed-point-spec.md` (which described `fixed-v1`; see "Changes from fixed-v1" below).

Authority order when documents disagree: the committed test vectors (`tests/vectors/phase6/`) > this document > the Python model (`scripts/phase5/fixed_detector.py`).
The vectors were produced by the vectorised model **and** by an independent oracle (`scripts/phase6/oracles.py`, plain ints / `Fraction`, no shared code); the generator aborts if the two disagree.
Do not change this contract silently. Any change requires a version bump, regenerated vectors (`scripts/phase6/generate_rtl_vectors.py`), a rerun of the float-vs-fixed comparison, and an entry in `reports/phase6/`.
Simplified variants (mean CAR, other guard-bit counts) are separate, named, benchmarked configurations and are **not** the frozen reference.

Notation: `>>` arithmetic right shift (floor); `sat24(v)` clamps to [−2^23, 2^23−1]; `rsr(v,b) = (v + 2^(b−1)) >> b` (round half up); all values two's complement.

## 1. ADC input format and scaling
* Native Neuropixels samples at **30 000 Hz**, AP gain 500, 10-bit ADC: 1 code = **2.34375 µV** (`75/32`). Container: signed 16-bit (`adc_code`).
* From calibrated volts (software only): `code = round_half_away_from_zero(v_µV / 2.34375)` saturated to [−32768, 32767]. In hardware the code is the ADC output; no conversion exists on-chip.
* Internal signal = `sat24(code << 10)` (**10 guard bits**, internal LSB = 2.34375/1024 = 0.0022888 µV). Codes beyond ±8191 saturate (`fixed-v1` omitted this saturation; native data never reach it).

## 2. Channel ordering and synchronization
* A *frame* is one sample of every channel at the same time index; ordering is **sample-major, channel-minor** (all channels of frame 0, then frame 1, …). Channel ids are dense `0..N−1`, `N ≤ 256` (reference tested at 1…33).
* The common reference needs a complete frame: the stage after the filter must collect all `N` filtered values of one time index before producing any CAR output. Skew between channels of a frame is not tolerated by the reference (an RTL frame assembler must guarantee it).
* `timestamp` is a u32 frame counter (wraps at 2^32 = 39.8 h; wrap handling is outside this contract).
* Per-channel `active` mask: inactive channels are filtered but excluded from the reference and never fire (threshold sentinel `−2^23`).
* Per-frame `valid`: a dropped frame is absent: filter state holds, no comparison happens, `timestamp` still advances, open peak searches still expire by timestamp, the "previous CAR value" used for crossing detection is the last present frame.

## 3. Common average reference (CAR)
Applied to the filter output (not the raw input), per frame, over the `n` active channels:
* `median` (frozen): sort the `n` s24 values; odd `n` → middle element; even `n` → `(lo + hi) >> 1` with `lo, hi` the two middle elements (floor; `(−3+0)>>1 = −2`).
* `mean` (variant): `ref = (Σx · R + 2^31) >> 32`, `R = (2^33 + n) // (2n)` (= round(2^32/n)); exact for power-of-two `n`, within 1 LSB of exact rounding otherwise.
* `none` (variant).
* Output `x − ref` saturated to s24. Median is 1-Lipschitz, so the integer median differs from the real median of the same inputs by at most the filter error.

## 4. Filter (per channel, cascade of two DF-I biquads)
```
acc  = b0*x0 + b1*x1 + b2*x2 − a1*y1 − a2*y2          (signed 64-bit; measured worst case 51 bits)
y0   = sat24( sign(acc) * ( |acc| >> 30 ) )            (truncate toward zero, saturate)
states: x1<=x0, x2<=x1, y1<=y0, y2<=y1 ; section 2 input = section 1 output; initial states 0
```
* Coefficients s32 Q2.30, `a0 = 2^30` implicit (from `scripts.fixed_point_model.quantize_sos`, round half away from zero). Input `x0` is s24.
* **Truncation toward zero is mandatory**: round-to-nearest leaves a persistent 0.595 µV (260 internal LSB) limit cycle after a burst; truncation decays to exactly 0.
* Frozen v2 coefficients (30 kHz), rows `b0 b1 b2 a0 a1 a2`:
  * High-pass 200 Hz: `1042404711 −2084809422 1042404711 1073741824 −2083894655 1011982365`
  * Low-pass 4500 Hz: `140774468 281548936 140774468 1073741824 −802932516 292288564`
  * Max quantised |pole| = 0.9708 (< 1). (Baseline 300/6000 Hz band: `scripts/phase5/fixed_detector.py:FixedSpec()` defaults.)
* Expected deviation from float64 (coefficient effect is zero; all deviation is truncation): worst case `< 1.47 µV` (analytic L1 bound), measured max 0.87 µV and RMS 0.12 µV on typical windows (0.51 µV RMS on one window dominated by a large shared artifact, where truncation error accumulates through the high-pass feedback); truncation toward zero adds a deterministic **+0.12 % gain** relative to float.

## 5. Noise estimate and threshold calibration (offline, deterministic integer)
Per channel on a calibration segment of CAR outputs `x` (the reference uses the evaluated window itself; thresholds calibrated on a different window of the same session gave the same F1 in Phase 5):
1. `b = imed(x)`, `m = imed(|x − b|)`, `imed` = integer median with even-count floor rule above.
2. Repeat 3×: `lim = (m · 582983) >> 16` (6σ in MAD units); keep `|x−b| ≤ lim` (all samples if fewer than 32 remain); recompute `b`, `m` on the kept samples.
3. `σq16 = m · 97164` (σ·2^16 internal LSB); `thr = clamp(b − rsr(1024 · σq16, 24), −2^23+1, 2^23−1)` (4.0σ; `1024` is 4.0 in Q8); `recip = 2^48 // σq16` (u32).
4. `m = 0` or inactive → `thr = −2^23`, `recip = 0` (channel disarmed). The constants are `round(65536/0.6744897502) = 97164` and `round(6·65536/0.6744897502) = 582983`.
Registers per channel: `baseline` s24, `thr` s24, `recip` u32, `armed` 1 bit (`mad`, `σq16` are calibration intermediates).

## 6. Detection and peak selection (per armed channel, present frame)
* Crossing at present frame `t` iff previous present CAR value `≥ thr` and current `< thr` (strict; equality is *not* below). No crossing on the first frame.
* Peak = minimum of CAR values over present frames with timestamp in `[t, t+15]` (0.5 ms); **first** minimum on ties.
* `delta = baseline − code` (> 0); `amp_q8 = min(0xFFFF, (delta · recip) >> 24)` (σ units, Q8.8).

## 7. Refractory, artifact veto, event merge
* Refractory (per channel, `last_peak` initial −31): ignore a crossing if `t_cross − last_peak < 30`; discard a peak if `t_peak − last_peak < 30` (exactly 30 apart is allowed).
* Artifact veto: sort candidates by (timestamp, channel); group anchored on the first within 6 samples; drop the whole group if it contains `≥ max(2, (128·n_active + 255) >> 8)` distinct channels.
* Merge: sort by (timestamp, channel); group anchored on the first within 15 samples; emit the member with the largest `amp_q8`, ties → earliest, then lowest channel.

## 8. Reset and initial state
All filter states 0; `last_peak = −31`; previous-value flags clear (no crossing on the first present frame); no open peak search. Calibration registers survive reset (they are programmed). A reset is required between finite recordings.

## 9. Buffering and end of stream
* Streaming needs no sample history beyond the filter states, one frame (N·24 b) for the reference, the previous CAR value per channel, and the open-search registers (timestamp, best timestamp, best value per channel).
* A peak search closes when frame timestamp `> t_cross + 15` (or at a dropped-frame/stream boundary check); candidates therefore appear up to 16 frames after the trough. Veto and merge hold-off add up to 6 + 15 frames. At end of stream `flush` closes every open search with the minimum seen so far; veto/merge then run on the remaining events.
* Backpressure: the reference is a pure function of the frame sequence; RTL may stall inputs or buffer events as long as the output sequence is identical (an event FIFO overflow must be counted and flagged; depth is a design choice, not fixed here).

## 10. Event output format
`(timestamp u32, channel u8, code s24, delta u25, amp_q8 u16)` per merged event, in nondecreasing timestamp order. `code` is the CAR output at the trough, `delta = baseline − code`, `amp_q8` the normalised amplitude. CSV in the vectors: `timestamp,channel,code,delta,amp_q8` (signed decimal).

## Test vectors (`tests/vectors/phase6/`, regenerate/verify with `scripts/phase6/generate_rtl_vectors.py [--check]`)
System scenarios (`nominal_32ch_median`, `nominal_8ch_mean_car`, `zero_input_4ch`, `extremes_4ch`, `adc_container_extremes_2ch`, `common_mode_8ch`, `dropped_frames_disabled_channels_8ch`, `overlapping_spikes_4ch`) each contain, in pipeline order:

| File | Content | Format |
|---|---|---|
| `adc_codes.hex` | input codes, present frames only | 16-bit hex, 1 word/line |
| `timestamps.txt` | u32 timestamp of each present frame | decimal |
| `section1_out.hex`, `section2_out.hex` | HP output, LP output (= filter output) | 24-bit hex |
| `car_ref.hex`, `car_out.hex` | per-frame reference; `sat24(x − ref)` | 24-bit hex |
| `config.json` | spec, active/valid masks, quantised coefficients, **calibration registers**, counts | JSON |
| `candidates.csv`, `candidates_after_veto.csv`, `events.csv` | per-channel candidates, after veto, final events | CSV |

Unit scenarios isolate each block: `detector_unit/` (equality, refractory ±1, stream start/end, tied minima, dropped frames, random small alphabet), `car_unit/` (1…32 channels, extremes, tie patterns; median and mean), `calibration_unit/` (Gaussian, outliers, exact `x.5` rounding tie, MAD = 0, tiny, offset), `merge_unit/` (veto and merge on 150 candidates).
**Debug order for RTL:** compare `section1_out` → `section2_out` → `car_ref` → `car_out` → calibration registers → `candidates` → `candidates_after_veto` → `events`; the first differing file names the first diverging stage.

## Hardware architecture assessment (estimates only; nothing synthesised)
| Block | Requirement | Estimate / alternatives |
|---|---|---|
| Median CAR, N = 32 | exact order statistic per frame | Pipelined Batcher odd-even sort: 191 compare-exchanges of 24 bit per frame; serial rank selection by bit slices: 24 passes × 32 comparisons ≈ 768 cycles. At the *assumed* 100 MHz and 30 kHz frames there are 3333 cycles/frame, so a serial selector fits with margin. Mean CAR: 31 adds + shift, no RAM, but measured as a different operating point (below). |
| Channel buffering | one frame (768 b) + in-flight frame; per-channel previous CAR value and open-search registers | tens of flip-flops per channel; no RAM for sample history |
| IIR arithmetic | 2 sections × 5 multiplies per channel-sample, 32×24-bit operands, 51-bit accumulator in practice | existing structure processes 1 channel-sample per 3 cycles (measured earlier on the 32 kHz RTL): 96 cycles/frame ≪ 3333. Narrowing the accumulator to 52 bits is possible; 64 is the verified margin. |
| Guard bits | 10 bits keep float/fixed agreement ≥ 98 % and worst-case error 1.47 µV | 12 bits would cut the bound to 0.37 µV at +2 accumulator bits; not adopted (not needed for detection quality). |
| Calibration | offline MAD/median, registers `{thr, baseline, recip, armed}` | ≈ 81 bits/channel (2.6 kb for 32 channels). On-chip streaming calibration is an alternative; a causal EMA noise estimate was inconclusive in Phase 5 selection (ΔF1 +0.009, CI includes 0; not adopted). |
| Veto + merge | hold-off ≤ 21 frames after the 16-frame search; reorder buffer for candidates emitted out of timestamp order | the existing merge module's DEPTH-64 queue is a starting point; tie rule must change to (timestamp, channel). |
| Backpressure | valid/ready on the sample port; event FIFO | stalls must not change the output sequence; overflow flagged, not silently dropped. |
| Cross-channel synchronisation | frame assembler before CAR | a hard requirement, absent from the current RTL. |

Measured effect of simplifying median → mean CAR (v2, 1 ms, channel-consistent F1, paired window bootstrap; Phase 6 `generalization.json`): development +0.002 (CI −0.020…+0.021), model-selection −0.008 (−0.020…+0.007), previously-used held-out **−0.015 (−0.025…−0.003)**, legacy −0.030; recall is lower by 4.5–6 points and false detections/s are 20–35 % lower; the sign of the F1 change depends on the session (better on A and C, worse on B and D). Keep median until a deliberate decision.

## Changes from `fixed-v1`
1. `MAD→σ` Q16 constant corrected 97157 → **97164** and the 6σ clip constant 582 964 → **582 983** (arithmetic slip found by re-deriving the constants; effect on events < 0.02 %).
2. The shifted ADC input is now saturated to s24 (v1 let codes beyond ±8191 enter the filter as >24-bit words).
3. Mean-CAR reciprocal computed with integers only (same values).
4. Version string `fixed-v1.1`. Native-range data produce the same events as `fixed-v1` to within the constant change (24 775 vs 24 777 exact events over 47 windows).
