# Phase 13 validation: visual information in spatial and temporal neural activity

Software only, public data only. The protocol was fixed on dev (`docs/phase13-visual-decoding.md`). Select and final
were each run once. The registry has 19 entries (`experiment_registry.jsonl`). Accuracy figures from different tasks
measure different things and are not compared with each other.

## Datasets (verified)

| Source | Sessions | Content used |
|---|---|---|
| Steinmetz session 11 (processed) | 1 | 32 VISp units, blank vs full right contrast, 167 train / 73 test trials |
| IBL insertions A-S | 19 checked; 18 usable (S has 5 visual units) | 402-850 trials, 50-451 visual-area units, 6 visual areas; contrasts 0 / 6.25 / 12.5 / 25 / 100% left or right |
| Allen Visual Coding Neuropixels (DANDI 000021) | 5 mice (one session each) | 396-952 good units in 5-6 visual areas; drifting gratings (8 orientations, 628 presentations), static gratings (6 orientations x 5 SF, 6,000), 118 natural images (5,950) |

Splits: IBL dev A-D, select E-L, final M-S (no final session was decoded in any earlier phase). Allen: dev = 2 mice,
select = 2, final = 1, by sorted session id. Only 5 Allen mice were streamed: DANDI downloads ran at ~0.5 MB/s, and
local disk was limited.

## Experiment A - reproduce simple classification

Steinmetz blank vs contrast reproduces exactly: 90.4% float and fixed-point accuracy against a 69.9% majority rate
(73 trials, Wilson 95% interval about 81-95%). The IBL contralateral-vs-ipsilateral side task (40-120 ms, unit counts,
balanced accuracy):

| Split | Counts | Shuffled labels | Pre-stimulus window | Still trials only | Choice decoding |
|---|---|---|---|---|---|
| dev (4) | 0.594 | 0.505 | 0.458 | 0.582 | 0.545 |
| select (8) | 0.579 | 0.486 | 0.483 | 0.560 | 0.529 |
| final (6) | 0.638 | 0.487 | 0.470 | 0.596 | **0.615** |

* Side information is above both chance controls in every split, and it survives on trials without wheel movement.
* In the final sessions, choice is also decodable (0.615), and error-trial side accuracy is mixed (0.38-0.64). Part of
  the final side signal may therefore be choice or movement related. This is a confound reported, not removed.
* **Stimulus presence (0% vs >= 25% contrast) is at chance in all splits (balanced accuracy 0.50-0.52)**, although the
  accuracy (~0.80) equals the majority rate. In this 40-120 ms window, most of these probes carry little presence
  information. Many sit in non-retinotopically matched or anterior areas (VISa, VISam).

## Experiment B - does precise timing help?

Time-binned counts against total counts, paired across sessions:

| Task | dev | select | final |
|---|---|---|---|
| IBL side | +0.5 pts (4/4) | +0.1 (3/8) | -0.5 (2/6) |
| **IBL contrast level (5 classes)** | +2.3 (3/4) | **+3.0 (7/8, p = 0.039)** | **+5.5 (5/6, CI +3.0/+7.8, p = 0.062)** |
| Allen static-grating orientation | +5.4, +3.8 | +5.1, +5.1 | +2.7 (CI +1.5/+4.0) |
| Allen spatial frequency | +3.1, +3.4 | +3.1, +4.8 | +3.8 |
| Allen drifting-grating orientation | +1.0, -2.2 | -2.3, -4.3 | 0.0 |
| Allen natural images (118) | +0.7, +4.2 | +0.6, -1.4 | +0.3 |

**Precise timing within the first 80-250 ms adds information about contrast, static-grating orientation and spatial
frequency.** The effect is reproducible across independent sessions, reaching +3 to +5 points. It does not add
information about drifting gratings, whose response is sustained.

## Experiment C - does localization help?

Spatial maps built from the same events and trials, with matched controls.

| Comparison | IBL (side / contrast) | Allen within-session (all tasks, 5 mice) |
|---|---|---|
| Depth map vs unit counts | -0.8 to -2.9 pts; final side -2.0 (1/6) | **-7 to -27 pts** in every session and task |
| Depth map vs dimension-matched random projection | select side **+3.7 (6/8, p = 0.039)**; final +2.5 (4/6, n.s.) | mixed (-8 to +5) |
| Depth map vs permuted-position map | +0.6 to +2.1, n.s. | -4.5 to +6.4; natural images +2.3 / +4.6 in 2 of 5 mice |
| Per-spike vs static unit depth | -3.1 (select, 0/8, p = 0.016); -1.7 final | n/a |
| Counts + map vs counts + permuted map | +0.1 to +0.6, n.s. | n/a |
| Spatio-temporal map vs time-binned counts | -2.1 to -4.6 | n/a |
| Matched MLP: spatio-temporal map vs time-binned counts | -1.7 to -3.9 | n/a |

* **Within a session, localization-based features never beat spike counts.** Units already carry their identity, and
  a depth map is a many-to-one pooling of the same events.
* Depth organization is not meaningless. In IBL side decoding, depth pooling beats an equally sized random pooling of
  units (select +3.7 pts), and natural-image decoding in two of five mice beats permuted positions. Neighbouring units
  share information, but less than the units themselves.
* **Per-spike (event-level, trial-varying) depth does not beat a fixed per-unit depth**; it is slightly worse.
* The P12 Allen unit-position comparison (`dmap_p12`) is **invalid**. My extraction mapped Allen's mean-waveform rows
  to the wrong electrodes: the waveform trough channel sits a median 86 um from the unit's peak channel. The result is
  reported as a defect, not as evidence about localization quality. The other Allen spatial representations use
  Allen's electrode positions and are unaffected.

## Experiment D - more detailed visual variables

Allen within-session, unit counts, balanced accuracy (chance in parentheses), dev / select / final:

* Drifting-grating orientation (8): 0.72-0.76 / 0.75 / 0.64 (0.125). Continuous orientation from (cos 2θ, sin 2θ):
  median error 2.8-4.7 deg with time bins.
* Static-grating orientation (6): 0.69-0.72 / 0.68-0.69 / 0.64 (0.167).
* Spatial frequency (5): 0.88-0.90 / 0.86-0.88 / 0.78 (0.20).
* Natural-image identity (118): 0.72-0.80 / 0.71-0.79 / 0.84 (0.008).
* IBL contrast level (5): 0.30-0.36 balanced accuracy (0.20 chance). Continuous log contrast R^2: 0.11-0.30.

This is image-identity classification among 118 presented images, not image reconstruction; no reconstruction claim
is made.

## Experiment E - does spatial geometry generalize across mice?

Decoders trained on other mice, using features defined identically across mice (VISp cortical-depth maps x time,
VISp total x time, per-area totals x time). All results are near chance:

| Task (chance) | dev to select (2 mice) | dev + select to final (1 mouse) |
|---|---|---|
| Drifting-grating orientation (0.125) | depth map 0.11 / 0.15; per-area 0.18 / 0.18 | depth 0.13; VISp total 0.17; per-area 0.15 |
| Static-grating orientation (0.167) | 0.15 / 0.20; per-area 0.22 / 0.18 | depth 0.16; per-area 0.21 |
| Natural images (0.008) | 0.013-0.020 | depth 0.010; per-area 0.029 |

Orientation in mouse visual cortex is salt-and-pepper, so cortical depth does not carry it across animals. Area-level
totals carry a little image information. **Spatial geometry does not provide a cross-mouse stimulus code here.**

## Section 11 - linking localization quality to decoding

* **Real data (IBL, per-spike depth with injected Gaussian noise; T2 balanced accuracy, sigma 0 / 10 / 25 / 50 / 100 /
  200 um):** dev 0.570 / 0.584 / 0.544 / 0.527 / 0.514 / 0.523; select 0.538 / 0.537 / 0.542 / 0.531 / 0.519 / 0.516;
  final 0.592 / 0.595 / 0.586 / 0.575 / 0.577 / 0.575. Map-based decoding loses 2-6 pts once position error reaches
  50-200 um, and is flat below about 25 um. The sorter's per-spike depth is already precise enough for what the map
  can use. The counts decoder, which needs no position, stays higher throughout.
* **Controlled simulation** (`sim13_localization_link.json`, `fig_sim_precision.png`; known positions, new neurons
  every session): cross-session decoding from depth maps collapses once unit-position error reaches about 1/4 to 1/2
  of the code's spatial period (20 um code: 0.97 at sigma 3 um, 0.65 at 12 um, chance at 25 um; 100 um code: 0.96 at
  25 um, 0.58 at 100 um). Event-level error behaves similarly. With a salt-and-pepper code, maps are at chance at any
  precision, while within-session counts reach 0.84-0.99. Phase 12's unit-level precision (~2-4 um, proxy) suffices
  for codes >= 20 um, and its single-event precision (~12 um) for codes >= 50 um.
* **Interpretation.** Localization helps decoding only when (a) neuron identities are not shared, as across sessions
  or animals, and (b) stimulus information is spatially organized at a scale larger than the position error. The real
  recordings tested here meet (a) across mice but not (b), which is why the cross-mouse maps are near chance. The
  shuffled-position and noise controls are what support this causal reading.

## Answers and targets

| Aim | Baseline | Result | Split | Reached |
|---|---|---|---|---|
| Reproduce simple baselines | Steinmetz 90.4% | 90.4% (identical) | - | yes |
| Timing improves over counts (>= 3 pts with consistency) | counts | contrast +3.0 (7/8) select, +5.5 (5/6) final; SG orientation +2.7 to +5.4 in every mouse | select, final | yes for contrast, SG orientation and SF; no for DG and side |
| Spatial maps improve over matched counts / temporal baselines (>= 3 pts) | counts, tcounts | -1 to -27 pts | all | no |
| Spatial maps beat shuffled positions | permuted map | +0.6 to +2.1 IBL (n.s.); mixed in Allen | all | no consistent effect |
| Spatial maps beat dimension-matched pooling | random projection | +3.7 IBL side (select, p = 0.039); not on final | select | partial, not replicated |
| Cross-session spatial generalization | chance | near chance | dev+select to final | no |
| Above majority and shuffled-label controls | - | yes for side, contrast, orientation, SF, images; no for presence | all | yes, except presence |

**A reliable negative result.** NeuroSig can decode stimulus side, contrast level, grating orientation, spatial
frequency and natural-image identity well above controls. Adding the spatial origin of events, as static unit
positions or per-event positions, does not improve decoding beyond spike counts or time-binned counts in any split or
dataset. The limit is the neural code (mouse visual-cortex tuning is not organized by probe depth at the measured
scales), not localization precision: degrading positions by up to ~25 um changes nothing, and simulation shows current
precision would suffice for a code organized at >= 20 um.

## Remaining limits

* Only 5 Allen mice. Single-mouse final and cross-mouse tests are therefore weak.
* IBL side decoding on the final sessions is partly confounded with choice.
* The Allen P12 position comparison must be redone with a correct channel map.
* Event-level positions from NeuroSig's own raw-voltage pipeline were not used for trial decoding: per-spike positions
  came from the sorter, and raw AP for every trial would be needed.
* Only 40-120 ms (IBL) and 0-250 ms (Allen) windows were tested.
