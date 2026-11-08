"""Phase 11B data: real trough-amplitude vectors from reliable events, footprint statistics, model comparison.

Training/validation events (selection rule fixed in advance): sorter events that are matched to a frozen
detector-v2.0 event (11A rule), re-centred with the E3 rule (raw-CAR multichannel argmin, +-3 samples), with
sorter-channel SNR >= 6 on the frozen preprocessing. Amplitude vector a_c = -(raw_car[s, c] - pre-event median);
sigma_c = MAD of raw_car. Low-SNR events are excluded from *training* only and are reported separately.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from edge.phase11 import forward_models as F
from edge.phase11.referencing import align_sample_raw, event_quality, load_events, match_ab, raw_car
from scripts.phase5.data import load_window, window_path

MIN_SNR = 6.0
WAVE = (-20, 40)


def collect(wins, min_snr=MIN_SNR, per_window=None, seed=0, keep_low=False):
    out = []
    rng = np.random.default_rng(seed)
    for wi, w in enumerate(wins):
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        we = load_events(w, load_window(p))
        rc, sig = raw_car(we)
        E = we.array.xyz_um()
        matched = {r for _, r, _ in match_ab(we, we.det_sample)}
        idx = sorted(matched)
        if per_window is not None and len(idx) > per_window:
            idx = sorted(rng.choice(idx, per_window, replace=False))
        for i in idx:
            s = align_sample_raw(we, int(we.sorter_sample[i]), int(we.sorter_channel[i]), 3)
            if s + WAVE[0] - 40 < 0 or s + WAVE[1] >= rc.shape[0]:
                continue
            q = event_quality(we, s, int(we.sorter_channel[i]))
            if q["snr"] < min_snr and not keep_low:
                continue
            base = np.median(rc[s - 40:s - 12], axis=0)
            wave = rc[s + WAVE[0]:s + WAVE[1]] - base[None]
            out.append({"session": w["session"], "win": wi, "unit": int(we.sorter_cluster[i]), "E": E,
                        "a": -wave[-WAVE[0]], "sigma": sig.copy(), "wave": wave, "snr": q["snr"],
                        "sorter_ch": int(we.sorter_channel[i])})
    return out


def line20_positions(evs, cache=None):
    m = F.Model("line20")
    return [F.fit(m, e["a"], e["sigma"], e["E"], cache=cache) for e in evs]


def footprint_samples(evs, fits):
    """Normalized amplitude vs in-plane offset from the line20 source estimate (for learning B/D)."""
    rad, ani = [], []
    for e, f in zip(evs, fits):
        A = float(f.amps[0])
        if A <= 0:
            continue
        dx = e["E"][:, 0] - f.theta[0]; dy = e["E"][:, 1] - f.theta[1]
        v = e["a"] / A
        rad.append((np.hypot(dx, dy), v)); ani.append((dx, dy, v))
    return rad, ani


def learn_models(evs, iterations=2, cache=None):
    """Learn B (radial) and D (aniso) from training events; positions re-estimated with the learned radial model."""
    fits = line20_positions(evs, cache)
    rad, ani = footprint_samples(evs, fits)
    radial = F.learn_radial(rad)
    for _ in range(iterations - 1):
        mb = F.Model("radial", radial=radial)
        fits = [F.fit(mb, e["a"], e["sigma"], e["E"], cache=cache) for e in evs]
        rad, ani = footprint_samples(evs, fits)
        radial = F.learn_radial(rad)
    aniso = F.learn_aniso(ani, radial)
    return F.Model("radial", radial=radial), F.Model("aniso", aniso=aniso)


def heldout_channel_error(model, evs, frac=0.25, seed=5, cache=None):
    """Fit on 75% of contacts, predict the held-out 25%: weighted relative error per event."""
    rng = np.random.default_rng(seed)
    errs, pos = [], []
    for e in evs:
        C = len(e["a"]); m = np.ones(C, bool); m[rng.choice(C, int(round(frac * C)), replace=False)] = False
        f = F.fit(model, e["a"], e["sigma"], e["E"], mask=m, cache=cache)
        pred = F.predict(model, f, e["E"]); w = 1.0 / np.square(e["sigma"])
        ho = ~m
        errs.append(float(np.sum(w[ho] * (e["a"][ho] - pred[ho]) ** 2) / max(np.sum(w[ho] * e["a"][ho] ** 2), 1e-12)))
    return np.array(errs)


def unit_spread(evs, fits, min_events=5):
    by = defaultdict(list)
    for e, f in zip(evs, fits):
        by[(e["session"], e["unit"])].append(f.theta[:2])
    out = {}
    for u, v in by.items():
        if len(v) >= min_events:
            a = np.array(v); out[u] = float(np.median(np.linalg.norm(a - np.median(a, axis=0), axis=1)))
    return out


def footprint_stats(evs, min_events=15):
    """Task 1 statistics per unit (mean waveform): polarity, axial asymmetry, temporal change of spatial extent,
    line20 template residual."""
    by = defaultdict(list)
    for e in evs:
        by[(e["session"], e["unit"])].append(e)
    rows = []
    m20 = F.Model("line20")
    for u, es in by.items():
        if len(es) < min_events:
            continue
        T = np.median(np.stack([e["wave"] for e in es]), axis=0)         # [t, C]
        sig = np.median(np.stack([e["sigma"] for e in es]), axis=0) / np.sqrt(len(es))
        E = es[0]["E"]; t0 = -WAVE[0]
        a = -T[t0]; pk = int(np.argmax(a))
        sigc = np.abs(T).max(axis=0) > 5 * sig
        pos_dom = (T.max(axis=0) > -T.min(axis=0)) & sigc
        f = F.fit(m20, a, sig * np.sqrt(len(es)), E)
        dy = E[:, 1] - E[pk, 1]; band = (np.abs(dy) >= 30) & (np.abs(dy) <= 70)
        up = a[band & (dy > 0)]; dn = a[band & (dy < 0)]
        def extent(row):
            v = np.abs(row); return float(np.sqrt(np.sum(v * (E[:, 1] - np.sum(v * E[:, 1]) / v.sum()) ** 2) / v.sum()))
        k_rep = t0 + int(np.argmax(T[t0:t0 + 20, pk]))
        rows.append({"session": u[0], "unit": int(u[1]), "n": len(es), "peak_uv": float(a[pk]),
                     "n_contacts_significant": int(sigc.sum()), "n_positive_dominant": int(pos_dom.sum()),
                     "axial_asymmetry_log_ratio": float(np.log((np.mean(up) + 1e-6) / (np.mean(dn) + 1e-6))) if len(up) and len(dn) and np.mean(up) > 0 and np.mean(dn) > 0 else None,
                     "extent_trough_um": extent(a), "extent_repol_um": extent(T[k_rep]),
                     "extent_pre_um": extent(T[max(t0 - 3, 0)]),
                     "width_ms": float(np.sum(-T[:, pk] > a[pk] / 2) / 30.0), "line20_rel_residual": f.sse_rel})
    return rows
