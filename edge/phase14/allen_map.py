"""Phase 14: pure helpers for the Allen (DANDI 000021) cache, kept free of I/O so they are unit-tested.

map_waveform_rows fixes the Phase 13 waveform-to-electrode mapping defect: a unit's ``waveform_mean`` block has one
row per channel of its probe ordered by the electrode's ``local_index``, while the NWB electrodes table is not stored
in local_index order. Phase 13 mapped row r to the r-th electrode of the probe in table order."""
from __future__ import annotations

import numpy as np


def map_waveform_rows(local_index: np.ndarray, probe_of_electrode: np.ndarray, peak_electrode: int) -> np.ndarray:
    """Electrode-table indices for the rows of a unit's waveform_mean block: row r -> electrode on the peak
    electrode's probe whose local_index == r. Returns -1 for rows with no such electrode."""
    same = np.flatnonzero(probe_of_electrode == probe_of_electrode[peak_electrode])
    li = np.asarray(local_index)[same]
    out = np.full(int(li.max()) + 1, -1, int)
    out[li] = same
    return out


def bin_counts(spikes: np.ndarray, onsets: np.ndarray, nbin: int = 50, bin_s: float = 0.005) -> np.ndarray:
    """[trials, nbin] counts of one unit's spikes in [on, on + nbin*bin_s) (vectorized; each spike is assigned to the
    latest onset at or before it, exact for non-overlapping windows)."""
    out = np.zeros((len(onsets), nbin), np.int32)
    order = np.argsort(onsets); on = np.asarray(onsets)[order]
    tr = np.searchsorted(on, spikes, side="right") - 1
    ok = tr >= 0
    rel = spikes[ok] - on[tr[ok]]
    b = np.floor(rel / bin_s).astype(int)
    k = (b >= 0) & (b < nbin)
    np.add.at(out, (order[tr[ok][k]], b[k]), 1)
    return out
