# Phase 13: visual information in spatial and temporal neural activity (software only)

Question: does the spatial organization of neural events carry stimulus information that a spike-count decoder
lacks? Phase 13 keeps decoding separate from localization. Every comparison uses identical trials, identical folds
and the same model family, plus shuffled-position, dimension-matched and shuffled-label controls. Results are in
`reports/phase13/phase13-validation.md`.

## Datasets (verified; inventories in `reports/phase13/`)

| Source | What is used | Ground truth | Inventory |
|---|---|---|---|
| Steinmetz et al. 2019, session 11 (processed) | 32 VISp unit counts | blank vs full right contrast | `steinmetz_repro.json` (Experiment A reproduction) |
| IBL Brain-Wide Map insertions A-S (the Phase 5-12 probes) | visual-area units, sorter unit depth, per-spike sorter depth, trials | stimulus side and contrast (0 / 6.25 / 12.5 / 25 / 100%), choice, first movement, block prior | `ibl_inventory.json` |
| Allen Brain Observatory Visual Coding Neuropixels (DANDI 000021, streamed) | good units in VIS* areas, probe positions, CCF, mean waveforms | drifting gratings (8 orientations), static gratings (6 orientations x 5 spatial frequencies), 118 natural images | `allen_inventory.json` |

Data stay out of Git (`data/phase13/`). Local disk was limited (~13 GB free), so only compact caches are kept: IBL
spikes of visual units only, and Allen per-presentation binned counts read by HTTP range requests (full-file
download measured at ~0.5 MB/s).

## Protocol (fixed on dev before select / final)

* IBL splits: dev A-D, select E-L, final M-S (no final session was decoded in any earlier phase). Window 40-120 ms
  after stimOn, 2 x 40 ms bins. On dev, the first wheel movement precedes +200 ms on 75-98% of trials, so movement
  exclusion is a control (still trials), not the main analysis. The other controls are a pre-stimulus window, choice
  decoding, error-trial accuracy and shuffled labels.
* Labels are hemisphere-aware: contralateral vs ipsilateral stimulus, from the probe's CCF ML coordinate.
* Allen: the first 2 sessions (sorted by id) are dev, the next 2 select, the rest final; every session is a different
  mouse. Window 0-250 ms, 5 x 50 ms bins. Blank presentations are excluded.
* Models: L2 logistic regression (C by inner CV on training folds only) and a matched small MLP; ridge regression
  for continuous targets. Folds are 5 contiguous blocks in time.
* Representations: unit counts; time-binned counts; total count; per-area totals; depth maps from sorter unit depth,
  peak-channel depth, per-spike depth, or P12 line20 unit positions (Allen); permuted-position maps;
  random projections of counts matched to map dimension; counts + map vs counts + permuted map; per-spike maps with
  injected position noise.
* Statistics: trial-bootstrap paired differences per session; across sessions the mean difference, sign count,
  session bootstrap interval and Wilcoxon p.

## Localization-to-decoding link (Section 11)

`scripts/phase13/run_sim13.py` is a controlled simulation with known positions. It sweeps the spatial scale of a
population code (20-250 um) and both unit-level and event-level position error, and includes a salt-and-pepper
control. On real data, the link is measured by injecting position noise into per-spike depth maps (IBL), by
comparing peak-channel with P12 positions (Allen), and by permuting positions.

## Reproduce

```bash
.venv/bin/python scripts/neuropixels_decoder.py --output data/phase13/steinmetz_repro --report reports/phase13/steinmetz_repro.json
data/ibl-venv/bin/python scripts/phase13/fetch_ibl13.py
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase13/run_ibl13.py dev        # then select, final (once)
data/allen-venv/bin/python scripts/phase13/fetch_allen13.py 5
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase13/run_allen13.py within dev   # select, final
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase13/run_allen13.py cross dev+select final
PYTHONPATH=. data/ibl-venv/bin/python scripts/phase13/run_sim13.py
```

`data/allen-venv` holds pynwb, remfile, h5py and dandi; analysis runs in `data/ibl-venv`, which has scikit-learn.
