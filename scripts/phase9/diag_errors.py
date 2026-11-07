"""Phase 3 diagnosis on development windows (sessions A-D, calibration + selection splits only)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge import ibl_eval
from edge.localize import LOCALIZATION_METHODS
from edge.localize.overlap import spatial_overlap_conflict
from edge.phase9.events import sorter_snippets
from scripts.phase5.data import load_window, window_path

m5 = json.load(open("reports/phase5/split_manifest.json"))["windows"]
wins = [w for w in m5 if w["split"] in ("calibration", "selection")]
rows = []
for w in wins:
    p = window_path(w["pid"], w["t0_s"], w["duration_s"])
    d = load_window(p)
    array, sn, mt = sorter_snippets(d, "probe", max_events=60, seed=int(w["t0_s"]))
    eng = ibl_eval._engines(array)
    for s, m in zip(sn, mt):
        v = s.voltages_uv; pk = m["local_channel"]
        amp_pk = -v[s.n_pre, pk]
        noise = np.median(np.abs(np.diff(v[:15], axis=0)), axis=0).mean() / 0.6745 / np.sqrt(2) + 1e-9
        row = {"snr": amp_pk / noise, "amp": amp_pk, "argmax_ch_y": float(array.xyz_um()[int(np.argmax(-v[s.n_pre])), 1]),
               "ref_y": m["ref_y_um"], "conflict": bool(spatial_overlap_conflict(s)[0]), "sess": w["pid"][:8]}
        for k in LOCALIZATION_METHODS:
            e = eng[k].estimate(s)
            row[k] = None if not e.available else abs(e.xyz_um[1] - m["ref_y_um"])
        rows.append(row)
snr = np.array([r["snr"] for r in rows])
print("events", len(rows), "snr pct 10/50/90", np.percentile(snr, [10, 50, 90]).round(1))
# peak-channel-of-trough vs sorter ref
d = np.array([abs(r["argmax_ch_y"] - r["ref_y"]) for r in rows]); print("argmax-trough channel vs sorter ref: frac 0um", np.mean(d == 0).round(3), "<=20", np.mean(d <= 20).round(3), "median", np.median(d))
bins = [(-9, 1), (1, 2), (2, 3), (3, 5), (5, 100)]
for lo, hi in bins:
    m = (snr > lo) & (snr <= hi)
    if m.sum() < 5: continue
    out = {k: (round(float(np.median([r[k] for r, mm in zip(rows, m) if mm and r[k] is not None])), 1)) for k in LOCALIZATION_METHODS}
    print(f"snr({lo},{hi}] n={m.sum():4d}", out, "robust-unavailable", round(float(np.mean([r['robust_scan'] is None for r, mm in zip(rows, m) if mm])), 2))
c = np.array([r["conflict"] for r in rows]); print("overlap-conflict fraction", c.mean().round(3), "(robust rejections are all from this rule)")
print("robust unavailable fraction", np.mean([r["robust_scan"] is None for r in rows]).round(3))
big = [r for r in rows if r["beamform"] is not None and r["beamform"] > 100]
print("beamform >100um:", len(big), "median snr", np.median([r["snr"] for r in big]).round(2), "frac argmax-ch matches ref within 20um", np.mean([abs(r['argmax_ch_y']-r['ref_y'])<=20 for r in big]).round(2))
