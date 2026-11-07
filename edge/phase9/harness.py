"""Real-data localization harness: conditioning, stability, dropout, noise. Proxies only (no neuron ground truth).

An *engine factory* maps an ElectrodeArray to {name: engine}. Everything is evaluated on identical events.
Dropout tests use engines rebuilt for the reduced contact set ("informed"); noise tests use the same engines.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from edge.phase9.events import sorter_snippets, window_array
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, ElectrodeSite, Snippet
from scripts.phase5.data import load_window, window_path
from scripts.real_detector import frozen_real_detector, preprocess

ROOT = Path(__file__).resolve().parents[2]


def manifest_windows(kind: str) -> list[dict]:
    """kind: 'dev' (Phase 5 calibration+selection, sessions A-D), 'heldout5' (Phase 5 heldout, used before),
    'locked' (Phase 6 locked_final, sessions E-H)."""
    m5 = json.loads((ROOT / "reports/phase5/split_manifest.json").read_text())
    m6 = json.loads((ROOT / "reports/phase6/split_manifest_phase6.json").read_text())
    sess = {v["pid"]: k for k, v in {**m5["sessions"], **m6["sessions"]}.items()}
    if kind == "dev":
        w = [x for x in m5["windows"] if x["split"] in ("calibration", "selection")]
    elif kind == "heldout5":
        w = [x for x in m5["windows"] if x["split"] == "heldout"]
    elif kind == "locked":
        w = list(m6["windows"])
    else:
        raise ValueError(kind)
    return [{**x, "session": sess[x["pid"]]} for x in w]


def condition_window(d: dict, mode: str) -> np.ndarray:
    if mode == "raw":
        return d["voltages_uv"]
    filt, _ = preprocess(d["voltages_uv"], frozen_real_detector(d["fs"]))
    return filt


def realign(v: np.ndarray, sample: int, half: int, local_ch: int | None, window_ch: int) -> int:
    """Move to the deepest trough within +-half samples on the strongest of the nearby channels (data-only)."""
    lo, hi = max(sample - half, 1), min(sample + half + 1, v.shape[0] - 1)
    seg = v[lo:hi]
    # strongest channel inside the window (not the sorter's); ties resolved toward the sorter channel's neighbourhood
    ch = int(np.unravel_index(int(np.argmin(seg)), seg.shape)[1])
    return lo + int(np.argmin(v[lo:hi, ch]))


def build_events(wins, conditioning: str = "raw", align_half: int = 0, max_events: int = 60, n_pre=None, n_post=None):
    """Return per-window dict(array, snippets, meta). Sorter times fix the nominal centre; align_half>0 re-centres on
    the data (global trough) within +-align_half samples."""
    out = []
    for w in wins:
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        d = load_window(p)
        v = condition_window(d, conditioning)
        array, sn, mt = sorter_snippets(d, "probe", max_events=max_events, seed=int(w["t0_s"]), voltages=v,
                                        n_pre=n_pre, n_post=n_post)
        if align_half > 0:
            sn2 = []
            for s, m in zip(sn, mt):
                new = realign(v, m["sample"], align_half, m["local_channel"], 0)
                kw = {}
                if n_pre is not None: kw["n_pre"] = n_pre
                if n_post is not None: kw["n_post"] = n_post
                sn2.append(extract_snippet(v, new, array, event_id=s.event_id, sample_rate_hz=s.sample_rate_hz,
                                           peak_channel=m["local_channel"], **kw))
                m["shift_samples"] = new - m["sample"]
            sn = sn2
        out.append({"window": w, "array": array, "snippets": sn, "meta": mt})
    return out


def _est(engine, sn):
    t0 = time.perf_counter()
    e = engine.estimate(sn)
    ms = (time.perf_counter() - t0) * 1e3
    ok = bool(e.available and e.xyz_um is not None)
    return (np.array(e.xyz_um) if ok else None), ms, e


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95))}


def sub_array(array: ElectrodeArray, keep: np.ndarray) -> ElectrodeArray:
    return ElectrodeArray(tuple(ElectrodeSite(i, array.sites[j].xyz_um) for i, j in enumerate(np.flatnonzero(keep))),
                          "reduced")


def sub_snippet(sn: Snippet, keep: np.ndarray) -> Snippet:
    idx = np.flatnonzero(keep)
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, tuple(range(len(idx))),
                   sn.voltages_uv[:, idx], sn.electrode_xyz_um[idx], None, None)


def make_mask(n: int, frac_keep: float, kind: str, rng) -> np.ndarray:
    k = int(round(n * frac_keep))
    if kind == "random":
        m = np.zeros(n, bool); m[rng.choice(n, k, replace=False)] = True
        return m
    start = int(rng.integers(0, n - (n - k) + 1))      # contiguous (spatially clustered) missing block
    m = np.ones(n, bool); m[start:start + (n - k)] = False
    return m


def perturb(sn: Snippet, kind: str, level: float, rng) -> Snippet:
    v = np.asarray(sn.voltages_uv, float).copy()
    d = np.diff(v, axis=0)
    sig = np.maximum(np.median(np.abs(d - np.median(d, axis=0)), axis=0) / 0.6745 / np.sqrt(2), 1e-6)
    T, C = v.shape
    if kind == "independent":
        v += rng.normal(size=v.shape) * (level * sig)[None, :]
    elif kind == "correlated":                        # one shared time series + neighbour smoothing across channels
        common = rng.normal(size=(T, 1)) * level * float(np.median(sig))
        v += common
    elif kind == "impulsive":                         # sparse large artifacts on random channels/samples
        n = max(1, int(0.02 * T * C))
        t = rng.integers(0, T, n); c = rng.integers(0, C, n)
        v[t, c] += rng.choice([-1, 1], n) * level * 6.0 * sig[c]
    elif kind == "gain":                              # per-channel multiplicative gain error
        v *= rng.uniform(1 - 0.3 * level, 1 + 0.3 * level, size=(1, C))
    elif kind == "drift":                             # slow baseline ramp + wobble per channel
        ramp = np.linspace(-1, 1, T)[:, None] * rng.normal(size=(1, C)) * level * 3.0 * sig[None, :]
        v += ramp
    else:
        raise ValueError(kind)
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v,
                   sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)


NOISE_CONDITIONS = (("independent", 1.0), ("correlated", 1.0), ("impulsive", 1.0), ("gain", 1.0), ("drift", 1.0))


def run(events, factory, seed=20269000, dropout=True, noise=True, max_drop_events=25):
    """Evaluate all engines from ``factory`` on ``events``. Returns a metric dict per method."""
    rng = np.random.default_rng(seed)
    rec = defaultdict(lambda: defaultdict(list))
    for ew in events:
        array, sns, mts = ew["array"], ew["snippets"], ew["meta"]
        engines = factory(array)
        sess = ew["window"]["session"]
        for i, (sn, mt) in enumerate(zip(sns, mts)):
            base = {}
            for name, eng in engines.items():
                xyz, ms, _ = _est(eng, sn)
                base[name] = xyz
                r = rec[name]
                r["ok"].append(xyz is not None); r["ms"].append(ms)
                r["axial"].append(None if xyz is None else abs(xyz[1] - mt["ref_y_um"]))
                if xyz is not None:
                    r["unit"].append((sess, mt["pid"], mt["cluster"], xyz[:2].copy(), mt["t0_s"], mt["ref_y_um"]))
            if dropout and i < max_drop_events:
                for kind in ("random", "clustered"):
                    for frac in (0.75, 0.5):
                        keep = make_mask(array.n_channels, frac, kind, rng)
                        red_engines = factory(sub_array(array, keep))
                        ssn = sub_snippet(sn, keep)
                        for name, eng in red_engines.items():
                            if base[name] is None:
                                continue
                            xyz, _, _ = _est(eng, ssn)
                            rec[name][f"drop_{kind}_{frac}"].append(None if xyz is None else float(np.linalg.norm(xyz[:2] - base[name][:2])))
            if dropout and i < max_drop_events:
                # unknown dropout: dead contacts read zero, engines still assume the full array
                for kind, frac in (("random", 0.75), ("random", 0.5), ("clustered", 0.75)):
                    keep = make_mask(array.n_channels, frac, kind, rng)
                    v0 = np.asarray(sn.voltages_uv, float).copy(); v0[:, ~keep] = 0.0
                    zsn = Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v0,
                                  sn.electrode_xyz_um, sn.peak_channel, sn.ground_truth_xyz_um)
                    for name, eng in engines.items():
                        if base[name] is None:
                            continue
                        xyz, _, _ = _est(eng, zsn)
                        rec[name][f"dropU_{kind}_{frac}"].append(None if xyz is None else float(np.linalg.norm(xyz[:2] - base[name][:2])))
            if noise and i < max_drop_events:
                for kind, lvl in NOISE_CONDITIONS:
                    psn = perturb(sn, kind, lvl, rng)
                    for name, eng in engines.items():
                        if base[name] is None:
                            continue
                        xyz, _, _ = _est(eng, psn)
                        rec[name][f"noise_{kind}"].append(None if xyz is None else float(np.linalg.norm(xyz[:2] - base[name][:2])))
    return summarize_records(rec)


def summarize_records(rec) -> dict:
    out = {}
    for name, r in rec.items():
        by_unit = defaultdict(list)
        refs = {}
        for sess, pid, cl, xy, t0, ref in r["unit"]:
            by_unit[(sess, pid, cl)].append((xy, t0))
            refs[(sess, pid, cl)] = ref
        spreads, per_sess, drifts = [], defaultdict(list), []
        med_pos, med_ref = [], []
        for (sess, pid, cl), v in by_unit.items():
            if len(v) < 5:
                continue
            med_pos.append(np.median(np.array([x for x, _ in v]), axis=0)); med_ref.append(refs[(sess, pid, cl)])
            a = np.array([x for x, _ in v]); sp = float(np.median(np.linalg.norm(a - np.median(a, axis=0), axis=1)))
            spreads.append(sp); per_sess[sess].append(sp)
            ts = np.array([t for _, t in v])
            if np.ptp(ts) > 0:
                c = np.median(ts); e, l = a[ts <= c], a[ts > c]
                if len(e) >= 2 and len(l) >= 2:
                    drifts.append(float(np.linalg.norm(np.median(e, axis=0) - np.median(l, axis=0))))
        from scipy.stats import spearmanr
        mp, mr = np.array(med_pos), np.array(med_ref)
        # rank correlation of unit-median depth with sorter peak-channel depth; computed within session so that
        # offsets between probes do not create correlation; a constant estimator scores 0
        rho_list, ratio_list = [], []
        sess_keys = sorted({k[0] for k in by_unit if len(by_unit[k]) >= 5})
        keys5 = [k for k in by_unit if len(by_unit[k]) >= 5]
        for sk in sess_keys:
            idx = [i for i, k in enumerate(keys5) if k[0] == sk]
            if len(idx) >= 6 and np.ptp(mp[idx, 1]) > 0:
                rho_list.append(float(spearmanr(mp[idx, 1], mr[idx]).statistic))
            if len(idx) >= 6:
                between = float(np.std(mp[idx, 1])); within = float(np.median([spreads[i] for i in idx]))
                ratio_list.append(within / max(between, 1e-9))
        res = {"unit_depth_rank_corr_median": float(np.median(rho_list)) if rho_list else None,
               "within_over_between_unit_spread": float(np.median(ratio_list)) if ratio_list else None,
               "valid_fraction": float(np.mean(r["ok"])), "runtime_ms_median": float(np.median(r["ms"])),
               "axial_disagreement_um": _stats(r["axial"]),
               "within_unit_spread_um": {"n_units": len(spreads), "median": float(np.median(spreads)) if spreads else None,
                                         "p90": float(np.percentile(spreads, 90)) if spreads else None},
               "per_session_median_spread_um": {s: float(np.median(v)) for s, v in sorted(per_sess.items())},
               "drift_um_median": float(np.median(drifts)) if drifts else None}
        for k in r:
            if k.startswith(("drop_", "dropU_", "noise_")):
                vals = r[k]
                st = _stats(vals)
                st["failure_fraction"] = float(np.mean([x is None for x in vals])) if vals else None
                res[k] = st
        out[name] = res
    return out
