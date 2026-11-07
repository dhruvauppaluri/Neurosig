"""Real dev windows: within-unit spread for refined variants vs beamform (axial MAD and xy spread)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from collections import defaultdict
from scripts.phase8.real_stability import *
from edge.phase8.refine import RefinedScanLocalizer

wins = windows_for("dev")[::2]   # every other window to save time
variants = {"beamform": None,
            "clip_nosvd": dict(clip_negative=True, use_temporal_svd=False),
            "clip_nosvd_prior30": dict(clip_negative=True, use_temporal_svd=False, prior_um=(40.0, 30.0)),
            "clip_nosvd_prior15": dict(clip_negative=True, use_temporal_svd=False, prior_um=(30.0, 15.0)),
            "svd_prior30": dict(clip_negative=True, prior_um=(40.0, 30.0)),
            "svd_prior15": dict(clip_negative=True, prior_um=(30.0, 15.0))}
store = {k: defaultdict(list) for k in variants}
for wi, w in enumerate(wins):
    path = window_path(w["pid"], w["t0_s"], w["duration_s"])
    array, snips, meta, win = events_from_window(path, 20268000 + wi)
    eng = baseline_engines(array)
    loc = {k: RefinedScanLocalizer(array, model="line_source", **kw) for k, kw in variants.items() if kw is not None}
    for sn, mt in zip(snips, meta):
        for k in variants:
            e = eng["beamform"].estimate(sn) if k == "beamform" else loc[k].estimate(sn)
            if e.available:
                store[k][(mt["pid"][:8], mt["cluster"])].append(e.xyz_um[:2])
for k, d in store.items():
    sp = [spread(np.array(v)) for v in d.values() if len(v) >= 5]
    ym = [np.median(np.abs(np.array(v)[:, 1] - np.median(np.array(v)[:, 1]))) for v in d.values() if len(v) >= 5]
    print(f"{k:22s} units={len(sp):3d} within-unit xy spread median={np.median(sp):6.1f}  axial MAD median={np.median(ym):6.1f}")
