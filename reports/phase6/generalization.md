# Detector generalization: baseline (300-6000 Hz) vs v2 (200-4500 Hz) vs v2 + mean CAR

Identical windows, identical sorter reference, identical matcher for every configuration; channel-consistent F1 at the stated tolerance (detection within 60 µm of the sorter's peak contact), with the shifted-reference chance level. Paired window-cluster bootstrap (2000 resamples) for differences.
Dataset roles: **development** (calibration windows: failure diagnostics, and where the narrower low-pass was first noticed, so not independent of v2), **model_selection** (v2 was chosen on these), **previously_used_heldout** (the Phase 5 'final' windows: evaluated once but since discussed; **not** a fresh holdout), **seen_legacy** (earlier-phase 1 s windows), **locked_final_phase6** (new, reserved before evaluation, evaluated once).

## development

12 of 12 reserved windows; sessions A, B, C.

| Config | Tol (ms) | F1 [95 % CI] | Recall | Precision | Chance F1 | False det./s | Unconstrained F1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.457 [0.431, 0.484] | 0.464 | 0.450 | 0.110 | 148 | 0.547 |
| baseline | 1.0 | 0.490 [0.461, 0.516] | 0.497 | 0.482 | 0.188 | 139 | 0.616 |
| baseline | 2.0 | 0.561 [0.528, 0.592] | 0.570 | 0.552 | 0.296 | 120 | 0.689 |
| v2 | 0.5 | 0.483 [0.454, 0.516] | 0.471 | 0.496 | 0.105 | 125 | 0.568 |
| v2 | 1.0 | 0.515 [0.486, 0.545] | 0.502 | 0.529 | 0.183 | 116 | 0.635 |
| v2 | 2.0 | 0.578 [0.546, 0.608] | 0.563 | 0.594 | 0.290 | 100 | 0.701 |
| v2_meancar | 0.5 | 0.486 [0.445, 0.526] | 0.427 | 0.564 | 0.096 | 86 | 0.564 |
| v2_meancar | 1.0 | 0.517 [0.472, 0.555] | 0.454 | 0.600 | 0.168 | 79 | 0.623 |
| v2_meancar | 2.0 | 0.569 [0.521, 0.608] | 0.500 | 0.660 | 0.268 | 67 | 0.675 |

| Paired difference | ΔF1 [95 % CI] | v2 better in windows | per session ΔF1 | Δrecall | Δprecision | false-det./s ratio |
|---|---|---|---|---|---|---|
| v2 − baseline @ 0.5 ms | +0.0260 [+0.0168, +0.0392] | 100 % | A +0.053, B +0.020, C +0.015 | +0.006 | +0.046 | 0.84 |
| v2 − baseline @ 1.0 ms | +0.0256 [+0.0173, +0.0378] | 100 % | A +0.050, B +0.017, C +0.019 | +0.004 | +0.047 | 0.83 |
| v2 − baseline @ 2.0 ms | +0.0171 [+0.0056, +0.0293] | 83 % | A +0.038, B +0.009, C +0.012 | -0.007 | +0.042 | 0.83 |
| mean-CAR − median-CAR (v2) @ 1.0 ms | +0.0015 [-0.0204, +0.0214] | 75 % | A +0.006, B -0.017, C +0.020 | -0.048 | +0.070 | 0.68 |

Windows where v2 F1 > baseline F1: 12 of 12 (worse: 0). Timing error of matched events (1 ms, channel-consistent): v2 median 0.033 ms, MAD 0.033 ms; baseline median 0.000 ms, MAD 0.033 ms. Per channel (channels with ≥ 20 reference spikes): v2 recall ≥ baseline on 33 of 54; median Δrecall +0.0000. Runtime (Python, total): baseline 2.0 s, v2 1.9 s, v2_meancar 1.6 s.

## model_selection

12 of 12 reserved windows; sessions A, B, C.

| Config | Tol (ms) | F1 [95 % CI] | Recall | Precision | Chance F1 | False det./s | Unconstrained F1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.416 [0.333, 0.476] | 0.433 | 0.400 | 0.097 | 179 | 0.540 |
| baseline | 1.0 | 0.455 [0.381, 0.512] | 0.474 | 0.438 | 0.167 | 168 | 0.612 |
| baseline | 2.0 | 0.518 [0.454, 0.569] | 0.540 | 0.499 | 0.273 | 149 | 0.690 |
| v2 | 0.5 | 0.444 [0.372, 0.500] | 0.447 | 0.442 | 0.094 | 156 | 0.552 |
| v2 | 1.0 | 0.480 [0.417, 0.530] | 0.483 | 0.477 | 0.163 | 146 | 0.627 |
| v2 | 2.0 | 0.538 [0.483, 0.583] | 0.541 | 0.535 | 0.270 | 130 | 0.694 |
| v2_meancar | 0.5 | 0.434 [0.353, 0.494] | 0.403 | 0.470 | 0.086 | 125 | 0.541 |
| v2_meancar | 1.0 | 0.472 [0.412, 0.522] | 0.438 | 0.511 | 0.152 | 116 | 0.612 |
| v2_meancar | 2.0 | 0.523 [0.470, 0.570] | 0.486 | 0.567 | 0.259 | 102 | 0.673 |

| Paired difference | ΔF1 [95 % CI] | v2 better in windows | per session ΔF1 | Δrecall | Δprecision | false-det./s ratio |
|---|---|---|---|---|---|---|
| v2 − baseline @ 0.5 ms | +0.0286 [+0.0170, +0.0444] | 100 % | A +0.058, B +0.014, C +0.016 | +0.014 | +0.042 | 0.87 |
| v2 − baseline @ 1.0 ms | +0.0246 [+0.0127, +0.0405] | 92 % | A +0.050, B +0.006, C +0.019 | +0.009 | +0.039 | 0.87 |
| v2 − baseline @ 2.0 ms | +0.0198 [+0.0065, +0.0365] | 83 % | A +0.045, B -0.002, C +0.018 | +0.002 | +0.036 | 0.87 |
| mean-CAR − median-CAR (v2) @ 1.0 ms | -0.0080 [-0.0202, +0.0067] | 42 % | A -0.007, B -0.031, C +0.016 | -0.044 | +0.034 | 0.79 |

Windows where v2 F1 > baseline F1: 11 of 12 (worse: 1). Timing error of matched events (1 ms, channel-consistent): v2 median 0.000 ms, MAD 0.033 ms; baseline median 0.000 ms, MAD 0.033 ms. Per channel (channels with ≥ 20 reference spikes): v2 recall ≥ baseline on 26 of 53; median Δrecall -0.0047. Runtime (Python, total): baseline 2.0 s, v2 1.9 s, v2_meancar 1.6 s.

## previously_used_heldout

20 of 20 reserved windows; sessions A, B, C, D.

| Config | Tol (ms) | F1 [95 % CI] | Recall | Precision | Chance F1 | False det./s | Unconstrained F1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.508 [0.481, 0.533] | 0.462 | 0.565 | 0.113 | 136 | 0.582 |
| baseline | 1.0 | 0.537 [0.511, 0.560] | 0.488 | 0.597 | 0.194 | 126 | 0.640 |
| baseline | 2.0 | 0.589 [0.562, 0.612] | 0.535 | 0.656 | 0.310 | 108 | 0.704 |
| v2 | 0.5 | 0.529 [0.501, 0.556] | 0.469 | 0.608 | 0.109 | 116 | 0.595 |
| v2 | 1.0 | 0.553 [0.527, 0.577] | 0.490 | 0.636 | 0.192 | 108 | 0.651 |
| v2 | 2.0 | 0.599 [0.573, 0.622] | 0.531 | 0.688 | 0.305 | 92 | 0.705 |
| v2_meancar | 0.5 | 0.519 [0.487, 0.549] | 0.421 | 0.675 | 0.098 | 78 | 0.573 |
| v2_meancar | 1.0 | 0.539 [0.509, 0.568] | 0.438 | 0.700 | 0.173 | 72 | 0.616 |
| v2_meancar | 2.0 | 0.575 [0.543, 0.606] | 0.467 | 0.747 | 0.278 | 61 | 0.657 |

| Paired difference | ΔF1 [95 % CI] | v2 better in windows | per session ΔF1 | Δrecall | Δprecision | false-det./s ratio |
|---|---|---|---|---|---|---|
| v2 − baseline @ 0.5 ms | +0.0209 [+0.0151, +0.0272] | 95 % | A +0.048, B +0.016, C +0.012, D +0.020 | +0.007 | +0.042 | 0.85 |
| v2 − baseline @ 1.0 ms | +0.0166 [+0.0117, +0.0224] | 95 % | A +0.043, B +0.014, C +0.011, D +0.014 | +0.002 | +0.038 | 0.85 |
| v2 − baseline @ 2.0 ms | +0.0098 [+0.0048, +0.0158] | 85 % | A +0.036, B +0.007, C +0.007, D +0.006 | -0.005 | +0.033 | 0.85 |
| mean-CAR − median-CAR (v2) @ 1.0 ms | -0.0148 [-0.0249, -0.0027] | 35 % | A +0.010, B -0.038, C +0.020, D -0.023 | -0.053 | +0.065 | 0.67 |

Windows where v2 F1 > baseline F1: 19 of 20 (worse: 1). Timing error of matched events (1 ms, channel-consistent): v2 median 0.000 ms, MAD 0.033 ms; baseline median 0.000 ms, MAD 0.033 ms. Per channel (channels with ≥ 20 reference spikes): v2 recall ≥ baseline on 53 of 84; median Δrecall +0.0000. Runtime (Python, total): baseline 3.3 s, v2 3.2 s, v2_meancar 2.6 s.

## seen_legacy

3 of 5 reserved windows; sessions A, B.

| Config | Tol (ms) | F1 [95 % CI] | Recall | Precision | Chance F1 | False det./s | Unconstrained F1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.554 [0.523, 0.577] | 0.499 | 0.623 | 0.130 | 94 | 0.639 |
| baseline | 1.0 | 0.585 [0.572, 0.596] | 0.527 | 0.658 | 0.216 | 85 | 0.697 |
| baseline | 2.0 | 0.639 [0.631, 0.647] | 0.576 | 0.718 | 0.339 | 70 | 0.754 |
| v2 | 0.5 | 0.580 [0.561, 0.595] | 0.512 | 0.668 | 0.127 | 79 | 0.652 |
| v2 | 1.0 | 0.602 [0.590, 0.611] | 0.532 | 0.693 | 0.220 | 73 | 0.717 |
| v2 | 2.0 | 0.657 [0.643, 0.676] | 0.580 | 0.756 | 0.332 | 58 | 0.758 |
| v2_meancar | 0.5 | 0.555 [0.542, 0.566] | 0.457 | 0.706 | 0.114 | 59 | 0.624 |
| v2_meancar | 1.0 | 0.572 [0.564, 0.582] | 0.471 | 0.728 | 0.197 | 55 | 0.678 |
| v2_meancar | 2.0 | 0.618 [0.604, 0.640] | 0.509 | 0.786 | 0.310 | 43 | 0.716 |

| Paired difference | ΔF1 [95 % CI] | v2 better in windows | per session ΔF1 | Δrecall | Δprecision | false-det./s ratio |
|---|---|---|---|---|---|---|
| v2 − baseline @ 0.5 ms | +0.0254 [+0.0090, +0.0377] | 100 % | A +0.018, B +0.038 | +0.013 | +0.045 | 0.84 |
| v2 − baseline @ 1.0 ms | +0.0163 [+0.0096, +0.0205] | 100 % | A +0.015, B +0.018 | +0.004 | +0.035 | 0.86 |
| v2 − baseline @ 2.0 ms | +0.0173 [+0.0109, +0.0290] | 100 % | A +0.020, B +0.013 | +0.004 | +0.038 | 0.83 |
| mean-CAR − median-CAR (v2) @ 1.0 ms | -0.0295 [-0.0350, -0.0262] | 0 % | A -0.032, B -0.026 | -0.060 | +0.035 | 0.75 |

Windows where v2 F1 > baseline F1: 3 of 3 (worse: 0). Timing error of matched events (1 ms, channel-consistent): v2 median 0.000 ms, MAD 0.033 ms; baseline median -0.033 ms, MAD 0.033 ms. Per channel (channels with ≥ 20 reference spikes): v2 recall ≥ baseline on 10 of 16; median Δrecall +0.0000. Runtime (Python, total): baseline 0.3 s, v2 0.3 s, v2_meancar 0.2 s.

## locked_final_phase6

24 of 24 reserved windows; sessions E, F, G, H.

| Config | Tol (ms) | F1 [95 % CI] | Recall | Precision | Chance F1 | False det./s | Unconstrained F1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.443 [0.399, 0.474] | 0.404 | 0.490 | 0.101 | 126 | 0.553 |
| baseline | 1.0 | 0.487 [0.447, 0.517] | 0.444 | 0.539 | 0.173 | 114 | 0.623 |
| baseline | 2.0 | 0.554 [0.513, 0.583] | 0.505 | 0.612 | 0.276 | 96 | 0.697 |
| v2 | 0.5 | 0.459 [0.414, 0.491] | 0.413 | 0.516 | 0.098 | 116 | 0.561 |
| v2 | 1.0 | 0.498 [0.455, 0.528] | 0.448 | 0.560 | 0.171 | 106 | 0.625 |
| v2 | 2.0 | 0.561 [0.521, 0.589] | 0.505 | 0.631 | 0.273 | 88 | 0.695 |
| v2_meancar | 0.5 | 0.453 [0.413, 0.482] | 0.371 | 0.581 | 0.092 | 80 | 0.545 |
| v2_meancar | 1.0 | 0.487 [0.449, 0.515] | 0.399 | 0.625 | 0.158 | 72 | 0.597 |
| v2_meancar | 2.0 | 0.539 [0.501, 0.566] | 0.441 | 0.691 | 0.253 | 59 | 0.650 |

| Paired difference | ΔF1 [95 % CI] | v2 better in windows | per session ΔF1 | Δrecall | Δprecision | false-det./s ratio |
|---|---|---|---|---|---|---|
| v2 − baseline @ 0.5 ms | +0.0163 [+0.0097, +0.0232] | 79 % | E +0.012, F -0.006, G +0.030, H +0.022 | +0.009 | +0.026 | 0.92 |
| v2 − baseline @ 1.0 ms | +0.0106 [+0.0022, +0.0183] | 83 % | E +0.008, F -0.016, G +0.026, H +0.017 | +0.004 | +0.021 | 0.93 |
| v2 − baseline @ 2.0 ms | +0.0075 [+0.0017, +0.0141] | 67 % | E +0.004, F -0.003, G +0.021, H +0.008 | +0.000 | +0.019 | 0.92 |
| mean-CAR − median-CAR (v2) @ 1.0 ms | -0.0108 [-0.0193, -0.0021] | 33 % | E -0.015, F +0.010, G -0.018, H -0.021 | -0.049 | +0.065 | 0.68 |

Windows where v2 F1 > baseline F1: 20 of 24 (worse: 4). Timing error of matched events (1 ms, channel-consistent): v2 median 0.000 ms, MAD 0.033 ms; baseline median 0.000 ms, MAD 0.033 ms. Per channel (channels with ≥ 20 reference spikes): v2 recall ≥ baseline on 45 of 70; median Δrecall +0.0084. Runtime (Python, total): baseline 4.0 s, v2 3.8 s, v2_meancar 3.2 s.

