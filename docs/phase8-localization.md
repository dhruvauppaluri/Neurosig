# Phase 8 localization: methods, protocol, limits

Software only (`edge/phase8/`). Results and goal status: `reports/phase8/phase8-validation.md`. The existing
`edge/localize/*` engines and their reports are unchanged.

## Components
| Module | Role |
|---|---|
| `refine.py` `RefinedScanLocalizer` | Fast K=1 tier (~1.3 ms): noise from first differences, whitened rank-1 temporal amplitudes, profile-likelihood grid, bounded LM. Not registered in `LOCALIZATION_METHODS`. |
| `multisource.py` `SpaceTimeFitter` | Variable-projection space-time fit with free temporal waveforms for K=1 and K=2; shared-waveform (rank-1) two-source test; Gauss-Newton covariance; `mask` for dropped contacts. |
| `pipeline.py` `Phase8Localizer` | Model-order selection, calibrated 90% regions, accept/reject with reason, `LocalizationResult` (`to_dict` keeps the legacy keys `available/method/xyz_um/confidence/reason/unverified`). `estimate()` returns the old `LocalizationEstimate`. |
| `pipeline.fit_calibration` | Calibrates thresholds and Mondrian split-conformal quantiles on the dev pool only. |
| `synth.py`, `overlap_bench.py` | Continuous-position generator with disjoint `dev/select/final` seeds; overlap benchmark with a fixed resolvable definition and Hungarian scoring. |
| `evaluate.py`, `registry.py` | Folded error metric, shared statistics, experiment registry (`reports/phase8/experiment_registry.jsonl`). |
| `final_eval.py`, `runtime_bench.py` | Single-shot final evaluation (refuses to rerun), latency/memory/burst benchmark. |
| `scripts/phase8/real_stability.py` | Repeated-unit, dropout, noise, drift, cross-session checks on cached IBL windows. |
| `edge/decode/phase8_ablation.py` | Counts/location/uncertainty decoding ablation across IBL insertions (network, `data/ibl-venv`). |

## Rules for using it
* Score collinear probes in the folded frame `(hypot(x,z), y)`; x and z are not identifiable. Use `edge.phase8.evaluate.folded_error_um`.
* Always report the accepted fraction beside any accuracy number; report region size with coverage.
* Calibration is valid for the monopole model it was fit on. Under forward-model mismatch coverage fell to 77.8%; recalibrate on a faithful model before relying on regions.
* On real recordings the continuous fit was less stable than beamform/template (`reports/phase8/phase8-validation.md`). Do not substitute it for the template recommendation without a new real-data protocol.
* Sorter-derived positions are proxies; do not optimize for agreement with the sorter peak channel.
* Final pools are single-use. `final_eval.py --force-rerun-final` is only for a documented new protocol.
* Neuropixels coordinates are probe-frame micrometers, not a stimulation targeting spec.

## Data split summary
Synthetic seeds: dev 20268001, select 20268002, final 20268003 (+7 for the staggered variant); overlap 20268101/2/3; real windows: sessions A–D for development, locked sessions E–H once; decode: all eight insertions, fixed hyperparameters, no tuning.
