"""Measure real spatial footprint and temporal width on development windows (strong, clean events only)."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.evalset import build_events_shift
from edge.phase9.improved import condition

wins = H.manifest_windows("dev")
ev = build_events_shift(wins, 0.0)
dist_amp, widths, n_used = [], [], 0
for ew in ev:
    for sn, mt in zip(ew["snippets"], ew["meta"]):
        c = condition(sn, baseline=True)
        v = c.voltages_uv; xyz = c.electrode_xyz_um; n = c.n_pre
        a = -v[n]
        pk = int(np.argmax(a))
        sig = float(np.median(np.abs(v[:10] - np.median(v[:10], axis=0))) / 0.6745)
        if a[pk] < 8 * sig or a[pk] < 40:
            continue
        n_used += 1
        r = np.linalg.norm(xyz - xyz[pk], axis=1)
        dist_amp.append(np.stack([r, a / a[pk]], axis=1))
        # temporal width at half-depth on the peak channel
        tr = -v[:, pk]; half = tr.max() / 2
        widths.append(float(np.sum(tr > half)) / sn.sample_rate_hz * 1000)
d = np.vstack(dist_amp)
print("strong events", n_used, "median FWHM-ish width ms", np.median(widths).round(3))
bins = [0, 1, 20, 30, 45, 60, 80, 110, 160, 250]
cent, med, p25, p75 = [], [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (d[:, 0] >= lo) & (d[:, 0] < hi)
    if m.sum() > 20:
        cent.append((lo+hi)/2); med.append(np.median(d[m, 1])); p25.append(np.percentile(d[m,1],25)); p75.append(np.percentile(d[m,1],75))
        print(f"r in [{lo},{hi}) n={m.sum():5d} median rel amp={med[-1]:.3f} IQR=({p25[-1]:.3f},{p75[-1]:.3f})")
cent=np.array(cent); med=np.array(med)
def sse(f): return float(np.sum((f - med)**2))
best = {}
for lam in (10,15,20,30,40,60,80):
    best[f"monopole 1/(r+L) L={lam}"] = sse((1/(cent+lam)) / (1/lam))
    best[f"line 1/sqrt(r^2+L^2) L={lam}"] = sse((1/np.sqrt(cent**2+lam**2)) / (1/lam))
for L in (20,30,40,60,80): best[f"exp(-r/L) L={L}"] = sse(np.exp(-cent/L))
for k, v in sorted(best.items(), key=lambda kv: kv[1])[:6]: print(f"{k:34s} sse={v:.3f}")
json.dump({"n_events": n_used, "width_ms": float(np.median(widths)), "r_um": cent.tolist(), "rel_amp_median": med.tolist()}, open("reports/phase9/real_footprint.json","w"), indent=1)
