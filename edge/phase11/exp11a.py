"""Phase 11A experiment core: localize every event of strategies A/B/C (+ +20 ms nulls) with identical engines and
summarize matched-event and full-stream metrics. Per-event records are returned for later decomposition."""

from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from edge.phase11.referencing import (ALIGN_HALF, align_sample, align_sample_raw, chance_matches, event_quality, load_events, match_ab,
                                      snippet_at)
from scripts.phase5.data import load_window, window_path

NULL_SHIFT_S = 0.020
B_DELAY_SAMPLES = 0      # median detector-minus-sorter offset on development windows (calibrated once, fixed)


def _loc(engines, sn):
    out = {}
    for name, e in engines.items():
        r = e.estimate(sn)
        out[name] = (np.array(r.xyz_um[:2]) if (r.available and r.xyz_um is not None) else None)
    return out


def run_windows(wins, factory, align_half=ALIGN_HALF, strategies=("A", "B", "C"), null=True, log=print,
                align_mode="filtered"):
    """Return (records, window_stats). A record: dict(strategy, null, session, win, unit, pair, sample, channel,
    snr, trough, xy={engine: xy|None}, ms)."""
    recs, wstats = [], []
    for wi, w in enumerate(wins):
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            wstats.append({"window": wi, "missing_dataset": True}); continue
        d = load_window(p)
        we = load_events(w, d)
        engines = factory(we.array)
        det_s = we.det_sample + B_DELAY_SAMPLES
        pairs = match_ab(we, det_s)
        a2b = {r: dd for dd, r, _ in pairs}
        b2a = {dd: r for dd, r, _ in pairs}
        t0 = time.perf_counter()
        fn = align_sample if align_mode == "filtered" else align_sample_raw
        c_s = np.array([fn(we, int(s), int(c), align_half) for s, c in zip(we.sorter_sample, we.sorter_channel)])
        align_ms = (time.perf_counter() - t0) * 1e3 / max(len(c_s), 1)
        wstats.append({"window": wi, "session": w["session"], "n_sorter": int(len(we.sorter_sample)),
                       "n_detector": int(len(det_s)), "n_matched": len(pairs), "chance_matched": chance_matches(we, det_s),
                       "dt_samples": [int(x[2]) for x in pairs], "c_shift_samples": (c_s - we.sorter_sample).tolist(),
                       "detector_runtime_s": we.det_runtime_s, "align_ms_per_event": align_ms})
        shift = int(round(NULL_SHIFT_S * we.fs))
        jobs = []
        if "A" in strategies:
            jobs += [("A", i, int(s), int(c), int(we.sorter_cluster[i]), i) for i, (s, c) in enumerate(zip(we.sorter_sample, we.sorter_channel))]
        if "C" in strategies:
            jobs += [("C", i, int(c_s[i]), int(we.sorter_channel[i]), int(we.sorter_cluster[i]), i) for i in range(len(c_s))]
        if "B" in strategies:
            jobs += [("B", j, int(s), int(c), (int(we.sorter_cluster[b2a[j]]) if j in b2a else None), b2a.get(j))
                     for j, (s, c) in enumerate(zip(det_s, we.det_channel))]
        for strat, idx, s, ch, unit, pair in jobs:
            for is_null in ((False, True) if null else (False,)):
                ss = s + (shift if is_null else 0)
                sn = snippet_at(we, ss, ch, idx)
                if sn is None:
                    continue
                q = event_quality(we, ss, ch)
                t = time.perf_counter()
                xy = _loc(engines, sn)
                recs.append({"strategy": strat, "null": is_null, "session": w["session"], "win": wi, "unit": unit,
                             "pair": pair, "matched": (pair is not None) if strat == "B" else (idx in a2b),
                             "sample": ss, "channel": ch, "snr": q["snr"], "trough": q["trough"], "xy": xy,
                             "ms": (time.perf_counter() - t) * 1e3})
        log(f"window {wi + 1}/{len(wins)} {w['session']} sorter {len(we.sorter_sample)} det {len(det_s)} matched {len(pairs)}")
    return recs, wstats


# ------------------------------------------------------------------------------------------------ summaries
def _unit_dev(rows, engine, min_events=5):
    """Per-event deviation from its unit median and per-unit median deviation (spread)."""
    by = defaultdict(list)
    for r in rows:
        xy = r["xy"].get(engine)
        if xy is not None and r["unit"] is not None:
            by[(r["session"], r["unit"])].append((r, xy))
    spreads, devs = {}, []
    for u, v in by.items():
        if len(v) < min_events:
            continue
        a = np.array([x for _, x in v]); dv = np.linalg.norm(a - np.median(a, axis=0), axis=1)
        spreads[u] = float(np.median(dv)); devs += list(zip([r for r, _ in v], dv))
    return spreads, devs


def _boot_ratio(sa: dict, sb: dict, seed=0, n=2000):
    keys = sorted(set(sa) & set(sb))
    if len(keys) < 3:
        return None
    a = np.array([sa[k] for k in keys]); b = np.array([sb[k] for k in keys])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), (n, len(keys)))
    r = np.median(b[idx], axis=1) / np.maximum(np.median(a[idx], axis=1), 1e-9) - 1
    return {"n_units": len(keys), "rel_change": float(np.median(b) / np.median(a) - 1),
            "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def summarize(recs, engines, wstats, fs=30000.0) -> dict:
    out = {"windows": len(wstats), "missing_dataset": sum(1 for w in wstats if w.get("missing_dataset"))}
    ws = [w for w in wstats if not w.get("missing_dataset")]
    ns, nd, nm, ch = (sum(w[k] for w in ws) for k in ("n_sorter", "n_detector", "n_matched", "chance_matched"))
    dts = np.concatenate([w["dt_samples"] for w in ws]) / fs * 1e3 if ws else np.zeros(0)
    cs = np.concatenate([w["c_shift_samples"] for w in ws]) / fs * 1e3 if ws else np.zeros(0)
    out["events"] = {"sorter": ns, "detector": nd, "matched": nm, "chance_matched": ch,
                     "matched_frac_of_sorter": nm / max(ns, 1), "matched_frac_of_detector": nm / max(nd, 1),
                     "chance_frac_of_sorter": ch / max(ns, 1),
                     "B_minus_A_ms": {"median": float(np.median(dts)), "iqr": [float(np.percentile(dts, 25)), float(np.percentile(dts, 75))]} if dts.size else None,
                     "C_minus_A_ms": {"median_abs": float(np.median(np.abs(cs))), "frac_moved": float(np.mean(cs != 0)),
                                       "frac_at_bound": float(np.mean(np.abs(cs) >= ALIGN_HALF / fs * 1e3 - 1e-9))},
                     "detector_runtime_ms_per_window": float(np.median([w["detector_runtime_s"] for w in ws]) * 1e3),
                     "align_ms_per_event": float(np.median([w["align_ms_per_event"] for w in ws]))}
    real = [r for r in recs if not r["null"]]; nul = [r for r in recs if r["null"]]
    per = {}
    for strat in ("A", "B", "C"):
        R = [r for r in real if r["strategy"] == strat]; N = [r for r in nul if r["strategy"] == strat]
        if not R:
            continue
        snr = np.array([r["snr"] for r in R])
        o = {"n_events": len(R), "n_labelled": sum(r["unit"] is not None for r in R),
             "snr_median": float(np.median(snr)), "trough_fraction": float(np.mean([r["trough"] for r in R])),
             "null_trough_fraction": float(np.mean([r["trough"] for r in N])) if N else None,
             "loc_ms_median": float(np.median([r["ms"] for r in R])), "engines": {}}
        for e in engines:
            sp, dv = _unit_dev(R, e); spn, _ = _unit_dev(N, e)
            d = np.array([x for _, x in dv]) if dv else np.zeros(0)
            o["engines"][e] = {"valid_fraction": float(np.mean([r["xy"][e] is not None for r in R])),
                               "spread_median": float(np.median(list(sp.values()))) if sp else None, "n_units": len(sp),
                               "dev_p90": float(np.percentile(d, 90)) if d.size else None,
                               "null_spread_median": float(np.median(list(spn.values()))) if spn else None}
        # method agreement: median pairwise distance between engines per event
        agr = []
        for r in R:
            P = [v for v in r["xy"].values() if v is not None]
            if len(P) >= 2:
                agr.append(np.median([np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]))
        o["method_disagreement_um_median"] = float(np.median(agr)) if agr else None
        per[strat] = o
    out["full_stream"] = per
    # matched-event analysis: identical sorter events that have a detector partner (A, C keyed by sorter index; B by pair)
    mk = lambda r: (r["win"], r["pair"] if r["strategy"] == "B" else None)
    keyA = {(r["win"], r["sample"] if False else None) for r in []}
    sets = {}
    for strat in ("A", "B", "C"):
        if strat == "B":
            rows = [r for r in real if r["strategy"] == "B" and r["pair"] is not None]
            keys = {(r["win"], r["pair"]) for r in rows}
        else:
            rows = [r for r in real if r["strategy"] == strat and r["matched"]]
        sets[strat] = rows
    # restrict A/C to sorter indices that have a B partner (A record's pair field is its own sorter index)
    bkeys = {(r["win"], r["pair"]) for r in sets["B"]}
    for strat in ("A", "C"):
        sets[strat] = [r for r in sets[strat] if (r["win"], r["pair"]) in bkeys]
    nulls = {s: [r for r in nul if r["strategy"] == s and ((r["win"], r["pair"]) in bkeys)] for s in ("A", "B", "C")}
    m = {"n_events": {s: len(v) for s, v in sets.items()}, "engines": {}}
    for e in engines:
        spA, dA = _unit_dev(sets["A"], e)
        row = {}
        for s in ("A", "B", "C"):
            sp, dv = _unit_dev(sets[s], e); spn, _ = _unit_dev(nulls[s], e)
            d = np.array([x for _, x in dv]) if dv else np.zeros(0)
            row[s] = {"spread_median": float(np.median(list(sp.values()))) if sp else None, "n_units": len(sp),
                      "dev_p90": float(np.percentile(d, 90)) if d.size else None,
                      "valid_fraction": float(np.mean([r["xy"][e] is not None for r in sets[s]])) if sets[s] else None,
                      "null_spread_median": float(np.median(list(spn.values()))) if spn else None,
                      "vs_A": _boot_ratio(spA, sp, seed=7) if s != "A" else None}
        m["engines"][e] = row
    out["matched"] = m
    # deviation by SNR quintile (strategy A, all labelled events, quintile edges fixed from A)
    out["dev_by_snr"] = {}
    for e in engines:
        _, dv = _unit_dev([r for r in real if r["strategy"] == "A"], e)
        if not dv:
            continue
        s = np.array([r["snr"] for r, _ in dv]); x = np.array([v for _, v in dv]); q = np.digitize(s, np.percentile(s, [20, 40, 60, 80]))
        out["dev_by_snr"][e] = [{"snr_median": float(np.median(s[q == k])), "dev_median": float(np.median(x[q == k])),
                                  "trough_frac": float(np.mean(s[q == k] >= 4.0))} for k in range(5)]
    return out
