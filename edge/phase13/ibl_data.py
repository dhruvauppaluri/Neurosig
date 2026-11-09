"""Phase 13 IBL data: compact per-session caches for visual-decoding experiments (run in data/ibl-venv).

For one insertion: the trials table (stimOn, contrasts, choice, feedback, first movement, block prior), every cluster
whose peak channel lies in a visual area (acronym starting 'VIS'), its sorter depth (clusters.depths), peak-channel
depth (electrode coordinate) and quality label, and the spikes of those clusters with per-spike depth
(spikes.depths, the sorter's per-spike position estimate along the probe) and amplitude. Probe hemisphere from the
channels' CCF ML coordinate (for contralateral / ipsilateral labels). Written to data/phase13/ibl_<key>.npz.
Large per-spike files fetched only for this cache (spikes.depths / spikes.amps) are deleted from the ONE cache after
compaction when ``prune`` is set (they can be re-downloaded), to respect local disk limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/phase13"


def fetch(key: str, pid: str, prune: bool = True) -> Path:
    out = CACHE / f"ibl_{key}.npz"
    if out.exists():
        return out
    from iblatlas.regions import BrainRegions
    from one.api import ONE
    from edge.ibl import SORTING_REVISION, TRIALS_REVISION
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
              silent=True, cache_dir=str(ROOT / "data/neuropixels/ibl_cache"))
    eid, probe = one.pid2eid(pid)
    coll = f"alf/{probe}/pykilosort"
    paths = []

    def ds(name, keep=True):
        p = one.load_dataset(eid, name, collection=coll, revision=SORTING_REVISION, download_only=True)
        if not keep:
            paths.append(Path(p))
        return np.load(p) if str(p).endswith(".npy") else p
    acr = BrainRegions().id2acronym(np.asarray(ds("channels.brainLocationIds_ccf_2017.npy"))).astype(str)
    coords = np.asarray(ds("channels.localCoordinates.npy"), float)
    mlapdv = np.asarray(ds("channels.mlapdv.npy"), float)
    cch = np.asarray(ds("clusters.channels.npy"), int)
    cdep = np.asarray(ds("clusters.depths.npy"), float)
    import pandas as pd
    met = pd.read_parquet(ds("clusters.metrics.pqt"))
    label = np.asarray(met["label"], float) if "label" in met.columns else np.zeros(len(cch))
    vis = np.flatnonzero(np.char.startswith(acr[cch].astype(str), "VIS"))
    st = np.asarray(ds("spikes.times.npy"), float)
    sc = np.asarray(ds("spikes.clusters.npy"), int)
    sd = np.asarray(ds("spikes.depths.npy", keep=False), float)
    sa = np.asarray(ds("spikes.amps.npy", keep=False), float)
    keep = np.isin(sc, vis)
    trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf", revision=TRIALS_REVISION)
    tcols = {c: np.asarray(trials[c], float) for c in ("stimOn_times", "contrastLeft", "contrastRight", "choice", "feedbackType",
                                                        "firstMovement_times", "probabilityLeft") if c in trials.columns}
    ml = float(np.nanmedian(mlapdv[cch[vis], 0])) if len(vis) else np.nan
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pid=pid, eid=str(eid), probe=probe, vis_clusters=vis, cluster_depth=cdep[vis],
                        cluster_peak_depth=coords[cch[vis], 1], cluster_label=label[vis], cluster_acronym=acr[cch[vis]],
                        spike_times=st[keep], spike_clusters=sc[keep], spike_depths=sd[keep].astype(np.float32),
                        spike_amps=sa[keep].astype(np.float32), hemisphere_ml_um=ml,
                        **{f"trial_{k}": v for k, v in tcols.items()})
    if prune:
        for p in paths:
            try:
                p.unlink()
            except OSError:
                pass
    return out


def load(key: str) -> dict:
    with np.load(CACHE / f"ibl_{key}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
