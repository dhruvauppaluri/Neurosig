# Float vs fixed comparison (all cached windows)

Two separate questions. **A. Numerical equivalence**: how closely does the integer model reproduce the float algorithm? **B. Detection quality**: does the integer model keep its agreement with the independent sorter reference after chance adjustment? A high float/fixed agreement is *not* evidence of biological detection correctness; the sorter is not ground truth.

## v2

| Scope | Windows | Float events | Fixed events | Exact (sample,channel) | ±3 samples same ch | ±15 samples same ch | Unmatched float @±15 | Float-only | Fixed-only | exact/float | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 47 | 25184 | 25146 | 24775 | 24917 | 24918 | 266 | 409 | 371 | 0.9838 | 0.9695 |
| session A | 14 | 5707 | 5711 | 5625 | 5650 | 5650 | 57 | 82 | 86 | 0.9856 | 0.9710 |
| session B | 13 | 5763 | 5756 | 5645 | 5684 | 5684 | 79 | 118 | 111 | 0.9795 | 0.9610 |
| session C | 12 | 8000 | 7975 | 7880 | 7918 | 7919 | 81 | 120 | 95 | 0.9850 | 0.9734 |
| session D | 8 | 5714 | 5704 | 5625 | 5665 | 5665 | 49 | 89 | 79 | 0.9844 | 0.9710 |

Amplitude differences on exactly matched events: normalised amplitude median 0.0121 σ, p99 0.053 σ, max 0.099 σ; trough value difference median 0.077 µV, p99 0.33 µV, max 0.55 µV.

| Scope | Float F1 | Fixed F1 | Fixed − float F1 [95 % CI] | Chance F1 (float / fixed) | Excess over chance (float / fixed) | Recall float / fixed | Precision float / fixed |
|---|---|---|---|---|---|---|---|
| all | 0.5283 | 0.5279 | -0.0004 [-0.0009, +0.0002] | 0.184 / 0.184 | 0.345 / 0.344 | 0.492 / 0.492 | 0.570 / 0.570 |
| session A | 0.5465 | 0.5467 | +0.0002 [-0.0008, +0.0011] | 0.138 / 0.139 | 0.408 / 0.408 | 0.571 / 0.571 | 0.524 / 0.524 |
| session B | 0.4942 | 0.4939 | -0.0003 [-0.0017, +0.0014] | 0.180 / 0.181 | 0.314 / 0.313 | 0.394 / 0.394 | 0.662 / 0.662 |
| session C | 0.4974 | 0.4971 | -0.0004 [-0.0012, +0.0005] | 0.210 / 0.210 | 0.287 / 0.287 | 0.575 / 0.573 | 0.439 / 0.439 |
| session D | 0.5834 | 0.5823 | -0.0010 [-0.0020, -0.0001] | 0.196 / 0.195 | 0.388 / 0.388 | 0.496 / 0.495 | 0.707 / 0.707 |
| split calibration | 0.5152 | 0.5149 | -0.0002 [-0.0013, +0.0011] | 0.183 / 0.183 | 0.332 / 0.332 | 0.502 / 0.501 | 0.529 / 0.529 |
| split heldout | 0.5534 | 0.5526 | -0.0008 [-0.0015, -0.0001] | 0.192 / 0.191 | 0.362 / 0.362 | 0.490 / 0.489 | 0.636 / 0.636 |
| split seen | 0.6017 | 0.6028 | +0.0011 [-0.0027, +0.0056] | 0.220 / 0.223 | 0.381 / 0.380 | 0.532 / 0.532 | 0.693 / 0.696 |
| split selection | 0.4799 | 0.4800 | +0.0001 [-0.0014, +0.0016] | 0.163 / 0.164 | 0.317 / 0.317 | 0.483 / 0.483 | 0.477 / 0.477 |

Threshold-boundary effects (candidate-level mismatches by first divergent stage): crossing_flip 765, peak_near_tie 380, refractory_cascade 381.

## baseline

| Scope | Windows | Float events | Fixed events | Exact (sample,channel) | ±3 samples same ch | ±15 samples same ch | Unmatched float @±15 | Float-only | Fixed-only | exact/float | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 47 | 26902 | 26881 | 26673 | 26734 | 26737 | 165 | 229 | 208 | 0.9915 | 0.9839 |
| session A | 14 | 5901 | 5902 | 5856 | 5865 | 5866 | 35 | 45 | 46 | 0.9924 | 0.9847 |
| session B | 13 | 5863 | 5849 | 5792 | 5810 | 5811 | 52 | 71 | 57 | 0.9879 | 0.9784 |
| session C | 12 | 9201 | 9193 | 9122 | 9146 | 9146 | 55 | 79 | 71 | 0.9914 | 0.9838 |
| session D | 8 | 5937 | 5937 | 5903 | 5913 | 5914 | 23 | 34 | 34 | 0.9943 | 0.9886 |

Amplitude differences on exactly matched events: normalised amplitude median 0.0064 σ, p99 0.027 σ, max 0.063 σ; trough value difference median 0.042 µV, p99 0.18 µV, max 0.32 µV.

| Scope | Float F1 | Fixed F1 | Fixed − float F1 [95 % CI] | Chance F1 (float / fixed) | Excess over chance (float / fixed) | Recall float / fixed | Precision float / fixed |
|---|---|---|---|---|---|---|---|
| all | 0.5076 | 0.5073 | -0.0002 [-0.0007, +0.0002] | 0.187 / 0.187 | 0.321 / 0.321 | 0.488 / 0.487 | 0.529 / 0.529 |
| session A | 0.5018 | 0.5014 | -0.0004 [-0.0016, +0.0008] | 0.141 / 0.141 | 0.361 / 0.360 | 0.533 / 0.533 | 0.474 / 0.473 |
| session B | 0.4813 | 0.4808 | -0.0005 [-0.0013, +0.0003] | 0.181 / 0.181 | 0.301 / 0.300 | 0.386 / 0.386 | 0.638 / 0.638 |
| session C | 0.4811 | 0.4814 | +0.0003 [-0.0006, +0.0010] | 0.219 / 0.219 | 0.262 / 0.263 | 0.603 / 0.603 | 0.400 / 0.401 |
| session D | 0.5699 | 0.5694 | -0.0004 [-0.0014, +0.0005] | 0.194 / 0.195 | 0.376 / 0.375 | 0.493 / 0.492 | 0.676 / 0.675 |
| split calibration | 0.4895 | 0.4894 | -0.0002 [-0.0011, +0.0007] | 0.188 / 0.188 | 0.302 / 0.301 | 0.497 / 0.497 | 0.482 / 0.482 |
| split heldout | 0.5368 | 0.5366 | -0.0002 [-0.0009, +0.0005] | 0.194 / 0.194 | 0.343 / 0.342 | 0.488 / 0.487 | 0.597 / 0.597 |
| split seen | 0.5854 | 0.5839 | -0.0015 [-0.0038, +0.0000] | 0.216 / 0.217 | 0.370 / 0.367 | 0.527 / 0.526 | 0.658 / 0.655 |
| split selection | 0.4553 | 0.4552 | -0.0002 [-0.0014, +0.0008] | 0.167 / 0.166 | 0.288 / 0.289 | 0.474 / 0.473 | 0.438 / 0.438 |

Threshold-boundary effects (candidate-level mismatches by first divergent stage): crossing_flip 443, peak_near_tie 175, refractory_cascade 176.

## v2+mean

| Scope | Windows | Float events | Fixed events | Exact (sample,channel) | ±3 samples same ch | ±15 samples same ch | Unmatched float @±15 | Float-only | Fixed-only | exact/float | Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 47 | 20608 | 20574 | 20395 | 20409 | 20414 | 194 | 213 | 179 | 0.9897 | 0.9811 |
| session A | 14 | 4807 | 4800 | 4743 | 4753 | 4757 | 50 | 64 | 57 | 0.9867 | 0.9751 |
| session B | 13 | 4543 | 4534 | 4473 | 4475 | 4475 | 68 | 70 | 61 | 0.9846 | 0.9715 |
| session C | 12 | 6584 | 6572 | 6535 | 6536 | 6537 | 47 | 49 | 37 | 0.9926 | 0.9870 |
| session D | 8 | 4674 | 4668 | 4644 | 4645 | 4645 | 29 | 30 | 24 | 0.9936 | 0.9885 |

Amplitude differences on exactly matched events: normalised amplitude median 0.0099 σ, p99 0.049 σ, max 0.113 σ; trough value difference median 0.063 µV, p99 0.27 µV, max 0.46 µV.

| Scope | Float F1 | Fixed F1 | Fixed − float F1 [95 % CI] | Chance F1 (float / fixed) | Excess over chance (float / fixed) | Recall float / fixed | Precision float / fixed |
|---|---|---|---|---|---|---|---|
| all | 0.5183 | 0.5179 | -0.0004 [-0.0013, +0.0004] | 0.167 / 0.167 | 0.351 / 0.351 | 0.442 / 0.442 | 0.626 / 0.626 |
| session A | 0.5429 | 0.5417 | -0.0012 [-0.0029, +0.0008] | 0.126 / 0.126 | 0.417 / 0.416 | 0.521 / 0.519 | 0.567 / 0.566 |
| session B | 0.4649 | 0.4643 | -0.0005 [-0.0027, +0.0018] | 0.160 / 0.160 | 0.305 / 0.305 | 0.341 / 0.341 | 0.728 / 0.728 |
| session C | 0.5159 | 0.5163 | +0.0003 [-0.0009, +0.0018] | 0.203 / 0.203 | 0.313 / 0.314 | 0.536 / 0.536 | 0.497 / 0.498 |
| session D | 0.5608 | 0.5603 | -0.0005 [-0.0012, +0.0000] | 0.173 / 0.173 | 0.388 / 0.387 | 0.441 / 0.441 | 0.769 / 0.769 |
| split calibration | 0.5167 | 0.5161 | -0.0006 [-0.0028, +0.0019] | 0.168 / 0.168 | 0.349 / 0.348 | 0.454 / 0.453 | 0.600 / 0.599 |
| split heldout | 0.5386 | 0.5377 | -0.0009 [-0.0017, -0.0001] | 0.173 / 0.173 | 0.365 / 0.365 | 0.438 / 0.436 | 0.700 / 0.700 |
| split seen | 0.5722 | 0.5731 | +0.0009 [-0.0026, +0.0058] | 0.197 / 0.197 | 0.376 / 0.376 | 0.471 / 0.473 | 0.728 / 0.728 |
| split selection | 0.4719 | 0.4723 | +0.0004 [-0.0016, +0.0026] | 0.152 / 0.152 | 0.320 / 0.321 | 0.438 / 0.438 | 0.511 / 0.512 |

