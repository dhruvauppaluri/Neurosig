"""Phase 10 real-data stability tests not run in Phase 9: recording segments, amplitude/SNR dependence, peak-electrode
removal. Proxies only (no neuron ground truth). Uses Phase 9 events (probe clock, sync_version>=3) and factories.

Segments: within each session the windows are ranked by start time and split into early / middle / late tertiles.
Per unit (>= 3 localized events in each of >= 2 tertiles) we report the largest distance between tertile medians and
the per-tertile within-unit spread. A null event set (same units, +20 ms) gives the floor these metrics reach
without any spike.

Amplitude: per event, trough depth on the sorter channel after pre-event baseline subtraction, divided by a robust
edge-noise estimate (SNR). Deviation of each estimate from its unit median is binned by SNR quintile (quintile edges
from the evaluated events, identical for every method).

Peak-electrode removal: the contact with the deepest trough (data-derived, not the sorter channel) is removed, either
known (engine rebuilt for the reduced array) or unknown (contact reads zero). Shift relative to the full estimate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from edge.phase9 import harness as H
from edge.phase9.improved import edge_noise
from edge.types import Snippet


def event_snr(sn: Snippet, channel: int) -> tuple[float, int]:
    v = np.asarray(sn.voltages_uv, float)
    k = max(5, sn.n_pre - 12)
    v = v - np.median(v[:k], axis=0, keepdims=True)
    lo, hi = max(sn.n_pre - 6, 0), min(sn.n_pre + 7, v.shape[0])
    noise = edge_noise(v, sn.n_pre)
    depth = -v[lo:hi].min(axis=0)
    peak = int(np.argmax(depth / noise))
    return float(depth[channel] / noise[channel]), peak


def _zero(sn: Snippet, ch: int) -> Snippet:
    v = np.asarray(sn.voltages_uv, float).copy(); v[:, ch] = 0.0
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um,
                   sn.peak_channel, sn.ground_truth_xyz_um)


def run_extra(events, factory, peak_drop=True, max_drop_events=25) -> dict:
    """Return {method: metrics}. ``events`` from edge.phase9.evalset.build_events_shift."""
    rows = defaultdict(list)            # method -> per-event dicts
    for ew in events:
        array, sns, mts = ew["array"], ew["snippets"], ew["meta"]
        engines = factory(array)
        sess = ew["window"]["session"]
        for i, (sn, mt) in enumerate(zip(sns, mts)):
            snr, peak = event_snr(sn, mt["local_channel"])
            red = None
            if peak_drop and i < max_drop_events:
                keep = np.ones(array.n_channels, bool); keep[peak] = False
                red = (factory(H.sub_array(array, keep)), H.sub_snippet(sn, keep), _zero(sn, peak))
            for name, eng in engines.items():
                xyz, _, _ = H._est(eng, sn)
                r = {"sess": sess, "unit": (sess, mt["pid"], mt["cluster"]), "t0": mt["t0_s"], "snr": snr,
                     "xy": None if xyz is None else xyz[:2].copy()}
                if red is not None and xyz is not None:
                    a, _, _ = H._est(red[0][name], red[1])
                    b, _, _ = H._est(eng, red[2])
                    r["peak_known"] = None if a is None else float(np.linalg.norm(a[:2] - xyz[:2]))
                    r["peak_unknown"] = None if b is None else float(np.linalg.norm(b[:2] - xyz[:2]))
                rows[name].append(r)
    return {m: summarize(r) for m, r in rows.items()}


def _tertiles(rows):
    by_sess = defaultdict(set)
    for r in rows:
        by_sess[r["sess"]].add(r["t0"])
    lab = {}
    for s, ts in by_sess.items():
        ts = sorted(ts)
        for k, t in enumerate(ts):
            lab[(s, t)] = min(int(3 * k / len(ts)), 2)
    return lab


def _st(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95))}


def summarize(rows) -> dict:
    lab = _tertiles(rows)
    ok = [r for r in rows if r["xy"] is not None]
    units = defaultdict(list)
    for r in ok:
        units[r["unit"]].append(r)
    seg_shift, seg_spread = [], {0: [], 1: [], 2: []}
    dev_snr, per_unit, per_sess = [], [], defaultdict(list)
    for u, rs in units.items():
        if len(rs) < 5:
            continue
        a = np.array([r["xy"] for r in rs]); med = np.median(a, axis=0)
        dev = np.linalg.norm(a - med, axis=1)
        per_unit.append(float(np.median(dev))); per_sess[u[0]].append(float(np.median(dev)))
        dev_snr += [(r["snr"], d) for r, d in zip(rs, dev)]
        seg = defaultdict(list)
        for r in rs:
            seg[lab[(r["sess"], r["t0"])]].append(r["xy"])
        meds = {k: np.median(np.array(v), axis=0) for k, v in seg.items() if len(v) >= 3}
        for k, v in seg.items():
            if len(v) >= 3:
                b = np.array(v); seg_spread[k].append(float(np.median(np.linalg.norm(b - np.median(b, axis=0), axis=1))))
        if len(meds) >= 2:
            ks = sorted(meds)
            seg_shift.append(max(float(np.linalg.norm(meds[i] - meds[j])) for i in ks for j in ks if i < j))
    out = {"valid_fraction": float(np.mean([r["xy"] is not None for r in rows])) if rows else None,
           "n_events": len(rows), "n_units_ge5": len(per_unit),
           "within_unit_spread_um": _st(per_unit),
           "per_session_median_spread_um": {s: float(np.median(v)) for s, v in sorted(per_sess.items())},
           "segment_shift_um": _st(seg_shift),
           "segment_spread_um": {name: _st(seg_spread[k]) for k, name in ((0, "early"), (1, "middle"), (2, "late"))}}
    if dev_snr:
        s = np.array(dev_snr)
        edges = np.percentile(s[:, 0], [20, 40, 60, 80])
        q = np.digitize(s[:, 0], edges)
        out["deviation_by_snr_quintile"] = [{"snr_median": float(np.median(s[q == k, 0])), **_st(s[q == k, 1])}
                                            for k in range(5)]
        from scipy.stats import spearmanr
        out["deviation_snr_spearman"] = float(spearmanr(s[:, 0], s[:, 1]).statistic)
    for key in ("peak_known", "peak_unknown"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[f"drop_{key}"] = {**_st(vals), "failure_fraction": float(np.mean([v is None for v in vals]))}
    return out
