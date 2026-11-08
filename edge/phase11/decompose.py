"""Phase 11A instability decomposition by controlled perturbation (dev windows, strategy-A sorter events).

Reference events: labelled sorter events with a measurable trough (SNR >= 8, i.e. clean spikes). For each:
  timing   shift the snippet centre by k samples (k = 1, 2, 3, 6, 10); displacement from the unshifted estimate.
  snr      mix the event snippet with a quiet (no-trough) snippet from the same window:  a*event + sqrt(1-a^2)*quiet,
           which keeps background power and scales the spike by a. Displacement vs realized SNR (a * SNR0).
Observed side: deviation from the unit median by SNR bin for all events, and overlap proxy (another sorter spike
within +-1 ms and 100 um). The predicted SNR curve is compared with the observed curve; the gap above the
prediction is attributed to factors other than noise (timing, overlap, model mismatch, reference errors).
Causal statements are restricted to the perturbed factor.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from edge.phase11.referencing import event_quality, load_events, snippet_at
from edge.types import Snippet
from scripts.phase5.data import load_window, window_path

TIMING_K = (1, 2, 3, 6, 10)
ALPHAS = (0.7, 0.5, 0.35, 0.25, 0.15)


def _mix(a: Snippet, b: Snippet, alpha: float) -> Snippet:
    va, vb = np.asarray(a.voltages_uv, float), np.asarray(b.voltages_uv, float)
    va = va - np.median(va[:20], axis=0); vb = vb - np.median(vb[:20], axis=0)
    v = alpha * va + np.sqrt(1 - alpha ** 2) * vb
    return Snippet(a.event_id, a.t_sample, a.t0_sample, a.sample_rate_hz, a.channel_ids, v, a.electrode_xyz_um,
                   a.peak_channel, None)


def run(wins, factory, engine="weighted_scan", min_snr=8.0, per_window=25, seed=11):
    rng = np.random.default_rng(seed)
    timing = defaultdict(list); snr_curve = []; overlap = []
    for w in wins:
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        we = load_events(w, load_window(p))
        eng = factory(we.array)[engine]
        E = we.array.xyz_um()
        # quiet snippets: random times with no trough (SNR < 2) on any contact set we use
        quiet = []
        for t in rng.integers(100, we.raw.shape[0] - 100, 400):
            q = event_quality(we, int(t), int(rng.integers(0, E.shape[0])))
            if q["snr"] < 2.0:
                quiet.append(int(t))
        qual = [event_quality(we, int(s), int(c)) for s, c in zip(we.sorter_sample, we.sorter_channel)]
        # overlap proxy over all sorter events
        for i, (s, c) in enumerate(zip(we.sorter_sample, we.sorter_channel)):
            near = np.abs(we.sorter_sample - s) <= int(0.001 * we.fs)
            near[i] = False
            ov = bool(np.any(near & (np.linalg.norm(E[we.sorter_channel] - E[c], axis=1) <= 100.0)))
            overlap.append((w["session"], int(we.sorter_cluster[i]), ov, qual[i]["snr"]))
        idx = [i for i, q in enumerate(qual) if q["snr"] >= min_snr]
        idx = list(rng.permutation(idx)[:per_window])
        for i in idx:
            s, c = int(we.sorter_sample[i]), int(we.sorter_channel[i])
            sn = snippet_at(we, s, c, i)
            if sn is None:
                continue
            r0 = eng.estimate(sn)
            if not r0.available:
                continue
            x0 = np.array(r0.xyz_um[:2])
            for k in TIMING_K:
                for sgn in (-1, 1):
                    sk = snippet_at(we, s + sgn * k, c, i)
                    if sk is None:
                        continue
                    r = eng.estimate(sk)
                    timing[k].append(np.linalg.norm(np.array(r.xyz_um[:2]) - x0) if r.available else np.nan)
            if quiet:
                qs = snippet_at(we, quiet[int(rng.integers(0, len(quiet)))], c, -1)
                if qs is None:
                    continue
                for a in ALPHAS:
                    r = eng.estimate(_mix(sn, qs, a))
                    snr_curve.append((a * qual[i]["snr"], np.linalg.norm(np.array(r.xyz_um[:2]) - x0) if r.available else np.nan))
    st = lambda v: {"n": int(np.isfinite(v).sum()), "median": float(np.nanmedian(v)), "p90": float(np.nanpercentile(v, 90))}
    out = {"engine": engine, "reference_min_snr": min_snr,
           "timing_shift_displacement_um": {str(k): st(np.array(v)) for k, v in timing.items()}}
    sc = np.array(snr_curve)
    bins = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 100)]
    out["snr_degradation_displacement_um"] = [{"snr_bin": b, **st(sc[(sc[:, 0] >= b[0]) & (sc[:, 0] < b[1]), 1])}
                                             for b in bins if np.any((sc[:, 0] >= b[0]) & (sc[:, 0] < b[1]))]
    ov = np.array([o[2] for o in overlap]); osnr = np.array([o[3] for o in overlap])
    out["overlap_proxy"] = {"fraction_events_with_neighbour_spike": float(ov.mean()),
                            "snr_median_overlap": float(np.median(osnr[ov])) if ov.any() else None,
                            "snr_median_isolated": float(np.median(osnr[~ov]))}
    out["_overlap_rows"] = overlap
    return out
