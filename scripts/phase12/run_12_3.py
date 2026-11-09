"""Phase 12.3 real-data part: single-event spatiotemporal variants on matched events. Usage: run_12_3.py dev|select

Population: matched sorter events (11A rule) of units with >= 5 matched events, at most 80 per unit (seeded), from the
session caches written by run_12_2.py. Every variant localizes the same events. Metrics follow Phase 11A: per-unit
median deviation from the unit median (spread), p90 of event deviations, valid fraction, fraction of deviations
> 50 um, and engine runtime. Paired unit-bootstrap relative change against the Phase 11 headline (ws_A: P11
weighted_scan on the unaligned sorter-time snippet).
"""
import argparse, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9.events import window_array
from edge.phase9.improved import condition
from edge.phase10.overlap_stress import localizer
from edge.phase11.configs import p11_factory
from edge.phase12 import pooling as PL, spatiotemporal as ST
from scripts.phase5.data import load_window, window_path
from edge.phase12.splits import split_windows

ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--per-unit", type=int, default=80)
a = ap.parse_args()
t0 = time.time()
by = defaultdict(list)
for w in split_windows(a.split):
    by[w["session"]].append(w)
rows = []
ms = defaultdict(list)
for sess, ws in sorted(by.items()):
    w0 = next(w for w in ws if window_path(w["pid"], w["t0_s"], w["duration_s"]).exists())
    array = window_array(load_window(window_path(w0["pid"], w0["t0_s"], w0["duration_s"])))
    p = Path(f"data/phase12/sess_{sess}.npz")
    if not p.exists():
        print("missing session cache", sess); continue
    se = PL.load_session(p, array)
    E = se.E; ws_eng = p11_factory(array)["weighted_scan"]; stl = localizer(array)
    sigma = np.median(se.sigma_rc, axis=0).astype(float)
    rng = np.random.default_rng(11)
    ev = []
    for u in np.unique(se.unit):
        idx = np.flatnonzero((se.unit == u) & se.matched)
        if len(idx) >= 5:
            ev += list(np.sort(rng.choice(idx, min(len(idx), a.per_unit), replace=False)))
    # per-window GLS fitter from event-free pre-event samples of that window's stored events
    gls = {}
    for wi in np.unique(se.win[ev]):
        j = np.flatnonzero(se.win == wi)[:200]
        rcs = [se.seg[i].astype(float) - np.median(se.seg[i].astype(float), axis=1, keepdims=True) for i in j]
        gls[wi] = ST.GLSLine20(E, ST.noise_cov(rcs, [-PL.SEG[0] + int(se.shift[i]) for i in j]))
    cache = {}
    for i in ev:
        seg = se.seg[i].astype(float); rc = seg - np.median(seg, axis=1, keepdims=True)
        k = -PL.SEG[0] + int(se.shift[i])
        W = ST.window(rc, k); Whp = ST.window(rc, k, hp=True)
        r = {"session": sess, "unit": int(se.unit[i]), "snr": float(se.snr[i]), "xy": {}}
        t = time.perf_counter(); e = ws_eng.estimate(se.snippet(i, aligned=False)); ms["ws_A"].append((time.perf_counter() - t) * 1e3)
        r["xy"]["ws_A"] = np.array(e.xyz_um[:2]) if e.available else None
        t = time.perf_counter(); e = ws_eng.estimate(se.snippet(i, aligned=True)); ms["ws_E3"].append((time.perf_counter() - t) * 1e3)
        r["xy"]["ws_E3"] = np.array(e.xyz_um[:2]) if e.available else None
        for m in ST.VARIANTS + ST.HP_VARIANTS:
            t = time.perf_counter()
            amp = ST.amplitudes(Whp if m.endswith("_hp") else W, sigma, m)
            r["xy"][m] = ST.localize_variant(m, amp, sigma, E, cache, gls[se.win[i]])
            ms[m].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        f = stl.fitter.fit(condition(se.snippet(i, aligned=True), baseline=True))
        ms["st_vp"].append((time.perf_counter() - t) * 1e3)
        r["xy"]["st_vp"] = None if f is None else np.array([f.sources1[0].x, f.sources1[0].y])
        a1 = ST.amplitudes(W, sigma, "rank1")
        r["ident_area"] = ST.identifiability_area(a1, sigma, E, cache)
        rows.append(r)
    print(sess, len(ev), f"{time.time() - t0:.0f}s", flush=True)

variants = list(rows[0]["xy"].keys())


def unit_dev(R, v):
    by = defaultdict(list)
    for r in R:
        if r["xy"][v] is not None:
            by[(r["session"], r["unit"])].append((r, r["xy"][v]))
    sp, dv = {}, []
    for u, L in by.items():
        if len(L) < 5:
            continue
        P = np.array([x for _, x in L]); d = np.linalg.norm(P - np.median(P, axis=0), axis=1)
        sp[u] = float(np.median(d)); dv += [(r, x) for (r, _), x in zip(L, d)]
    return sp, dv


def boot(sa, sb, seed=7, n=2000):
    k = sorted(set(sa) & set(sb)); A = np.array([sa[x] for x in k]); B = np.array([sb[x] for x in k])
    rng = np.random.default_rng(seed); I = rng.integers(0, len(k), (n, len(k)))
    rr = np.median(B[I], axis=1) / np.median(A[I], axis=1) - 1
    return {"n_units": len(k), "rel_change": float(np.median(B) / np.median(A) - 1),
            "ci95": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]}


def p90_boot(da, db, seed=8, n=1000):
    A = np.array([x for _, x in da]); B = np.array([x for _, x in db]); rng = np.random.default_rng(seed)
    rr = [np.percentile(B[rng.integers(0, len(B), len(B))], 90) / np.percentile(A[rng.integers(0, len(A), len(A))], 90) - 1 for _ in range(n)]
    return {"rel_change": float(np.percentile(B, 90) / np.percentile(A, 90) - 1), "ci95": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]}


spA, dA = unit_dev(rows, "ws_A")
res = {"versions": versions(), "split": a.split, "n_events": len(rows), "variants": {}, "runtime_s": None}
for v in variants:
    sp, dv = unit_dev(rows, v)
    d = np.array([x for _, x in dv])
    snr = np.array([r["snr"] for r, _ in dv])
    res["variants"][v] = {"spread_median": float(np.median(list(sp.values()))), "n_units": len(sp),
                          "dev_p90": float(np.percentile(d, 90)), "dev_gt50_frac": float(np.mean(d > 50)),
                          "valid_fraction": float(np.mean([r["xy"][v] is not None for r in rows])),
                          "ms_median": float(np.median(ms[v])), "vs_ws_A": boot(spA, sp), "p90_vs_ws_A": p90_boot(dA, dv),
                          "dev_median_by_snr": {g: float(np.median(d[(snr >= lo) & (snr < hi)])) if np.any((snr >= lo) & (snr < hi)) else None
                                                for g, (lo, hi) in {"lt8": (0, 8), "8to16": (8, 16), "ge16": (16, 1e9)}.items()},
                          "per_session_spread": {s: float(np.median([x for (ss, _), x in sp.items() if ss == s]))
                                                 for s in sorted({k[0] for k in sp})}}
# identifiability: does the profile-likelihood area predict the observed deviation of the rank1 estimate?
_, dv = unit_dev(rows, "rank1")
ar = np.array([r["ident_area"] for r, _ in dv]); dd = np.array([x for _, x in dv])
qq = np.minimum((np.argsort(np.argsort(ar, kind="stable")) * 4) // len(ar), 3)        # rank quartiles (ties split)
res["identifiability"] = {"spearman_area_vs_dev": float(spearmanr(ar, dd).statistic), "area_median_um2": float(np.median(ar)),
                          "dev_median_by_area_quartile": [float(np.median(dd[qq == q])) for q in range(4)]}
res["runtime_s"] = time.time() - t0
Path(f"reports/phase12/exp12_3_{a.split}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
pickle.dump(rows, open(f"data/phase12/rows12_3_{a.split}.pkl", "wb"))
for v, o in res["variants"].items():
    print(f"{v:13s} spread {o['spread_median']:5.1f} p90 {o['dev_p90']:5.1f} >50 {o['dev_gt50_frac']:.3f} valid {o['valid_fraction']:.3f} "
          f"ms {o['ms_median']:.2f} vsA {o['vs_ws_A']['rel_change']:+.3f} {o['vs_ws_A']['ci95']} p90 {o['p90_vs_ws_A']['rel_change']:+.3f}")
print(res["identifiability"])
