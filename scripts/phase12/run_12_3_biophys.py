"""Phase 12.3 model-mismatched check: the same single-event variants on biophysical templates + real noise.

Usage: run_12_3_biophys.py dev|select. Template and noise pools per edge/phase12/biosim.py. For every template:
SNR 4/8/16 (peak trough over the 300-6000 Hz filtered noise MAD of its peak contact, the 11A SNR measure) x 8 noise draws, reference time jittered by
U{-2..2} samples (the sorter/detector timing uncertainty seen in 11A), then the E3 rule (multichannel argmin over
contacts within 60 um of the reference-sample peak contact, +-3 samples). Errors are in-plane distances to the soma
(truth known); precision = per (template, SNR) median distance of the 8 estimates from their median (the simulated
analogue of within-unit spread); bias = distance of that median from the soma.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9.improved import condition
from edge.phase10.overlap_stress import localizer
from edge.phase11.configs import p11_factory
from edge.phase12 import biosim as B, spatiotemporal as ST
from edge.types import ElectrodeArray, ElectrodeSite, Snippet

pool = sys.argv[1]
WHICH = B.tset(pool)
rng = np.random.default_rng({"dev": 121, "select": 122, "final": 123}[pool])
nb = B.noise_bank(pool); sig_noise = B.noise_sigma(pool); sig_fit = B.noise_mad(nb)
t0 = time.time()
rows = []
ms = defaultdict(list)
engines = {}
for k in B.template_pool(pool):
    W, E, truth, tt, pk = B.template_view(k, WHICH)
    key = E.tobytes()
    if key not in engines:
        arr = ElectrodeArray(tuple(ElectrodeSite(i, tuple(E[i])) for i in range(32)), "biophys")
        nz = nb[:300].astype(float)
        gl = ST.GLSLine20(E, ST.noise_cov([x - np.median(x, axis=1, keepdims=True) for x in nz], [B.CENTER] * len(nz)))
        engines[key] = (arr, p11_factory(arr)["weighted_scan"], localizer(arr), {}, gl)
    arr, ws, stl, cache, gls = engines[key]
    for snr in (4.0, 8.0, 16.0):
        for rep in range(8):
            ev = B.make_single(k, snr, nb, sig_noise, rng, which=WHICH)
            seg = ev.seg; rc = seg - np.median(seg, axis=1, keepdims=True)
            sig = np.median(np.abs(rc - np.median(rc, axis=0)), axis=0) / 0.6745
            sig = np.maximum(sig_fit, 1e-6)          # raw-CAR noise level known from the bank (a recording-level MAD, as in real data)
            s = ev.ref_sample
            ch = int(np.argmin(rc[s]))
            S = np.flatnonzero(np.linalg.norm(E[:, :2] - E[ch, :2], axis=1) <= 60.0)
            base = np.median(rc[s - 40:s - 12][:, S], axis=0)
            sc = ((rc[s - 3:s + 4][:, S] - base) / sig[S]).sum(axis=1)
            s3 = s - 3 + int(np.argmin(sc))
            Wn = ST.window(rc, s3); Whp = ST.window(rc, s3, hp=True)
            r = {"template": k, "snr": snr, "rep": rep, "truth": truth, "xy": {}, "jitter": s - B.CENTER, "align_err": s3 - B.CENTER}
            for name, sample in (("ws_ref", s), ("ws_E3", s3)):
                sn = Snippet(0, 40, s - 40, 30000.0, tuple(range(32)), seg[sample - 40:sample + 60], E, ch, None)
                t = time.perf_counter(); e = ws.estimate(sn); ms[name].append((time.perf_counter() - t) * 1e3)
                r["xy"][name] = np.array(e.xyz_um[:2]) if e.available else None
            for m in ST.VARIANTS + ST.HP_VARIANTS:
                t = time.perf_counter()
                a = ST.amplitudes(Whp if m.endswith("_hp") else Wn, sig, m)
                r["xy"][m] = ST.localize_variant(m, a, sig, E, cache, gls)
                ms[m].append((time.perf_counter() - t) * 1e3)
            sn = Snippet(0, 40, s3 - 40, 30000.0, tuple(range(32)), seg[s3 - 40:s3 + 60], E, ch, None)
            t = time.perf_counter(); f = stl.fitter.fit(condition(sn, baseline=True)); ms["st_vp"].append((time.perf_counter() - t) * 1e3)
            r["xy"]["st_vp"] = None if f is None else np.array([f.sources1[0].x, f.sources1[0].y])
            r["ident_area"] = ST.identifiability_area(ST.amplitudes(Wn, sig, "rank1"), sig, E, cache)
            rows.append(r)
    print(k, f"{time.time() - t0:.0f}s", flush=True)

out = {"versions": versions(), "pool": pool, "n_templates": len(B.template_pool(pool)), "n_events": len(rows), "variants": {}}
for v in rows[0]["xy"]:
    o = {}
    for snr in (4.0, 8.0, 16.0):
        R = [r for r in rows if r["snr"] == snr]
        err = np.array([np.linalg.norm(r["xy"][v] - r["truth"]) if r["xy"][v] is not None else np.nan for r in R])
        prec, bias = [], []
        by = defaultdict(list)
        for r in R:
            if r["xy"][v] is not None:
                by[r["template"]].append(r["xy"][v])
        for kk, P in by.items():
            if len(P) >= 4:
                P = np.array(P); m = np.median(P, axis=0)
                prec.append(float(np.median(np.linalg.norm(P - m, axis=1))))
                bias.append(float(np.linalg.norm(m - R[[r["template"] for r in R].index(kk)]["truth"])))
        o[f"snr{int(snr)}"] = {"err_median": float(np.nanmedian(err)), "err_p90": float(np.nanpercentile(err, 90)),
                               "valid": float(np.mean(np.isfinite(err))), "precision_median": float(np.median(prec)),
                               "bias_median": float(np.median(bias)), "err_gt50": float(np.nanmean(err > 50))}
    o["ms_median"] = float(np.median(ms[v]))
    out["variants"][v] = o
from scipy.stats import spearmanr
err = np.array([np.linalg.norm(r["xy"]["rank1"] - r["truth"]) if r["xy"]["rank1"] is not None else np.nan for r in rows])
ar = np.array([r["ident_area"] for r in rows]); ok = np.isfinite(err)
out["identifiability"] = {"spearman_area_vs_err": float(spearmanr(ar[ok], err[ok]).statistic)}
out["runtime_s"] = time.time() - t0
Path(f"reports/phase12/exp12_3_biophys_{pool}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
for v, o in out["variants"].items():
    print(f"{v:13s} " + " | ".join(f"snr{s}: err {o[f'snr{s}']['err_median']:5.1f}/{o[f'snr{s}']['err_p90']:5.1f} prec {o[f'snr{s}']['precision_median']:4.1f} bias {o[f'snr{s}']['bias_median']:4.1f}" for s in (4, 8, 16)) + f" ms {o['ms_median']:.2f}")
print(out["identifiability"])
