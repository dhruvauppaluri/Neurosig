"""Phase 11B on development windows (A-C): footprint study, leave-one-session-out model comparison
(held-out-contact prediction, in-sample residual, within-unit spread, dropout displacement, runtime), plus the
independent biophysical (MEArec/NEURON) templates. Writes reports/phase11/exp11b_dev.json and caches events."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase11 import forward_models as F
from edge.phase11.footprint import (collect, footprint_stats, heldout_channel_error, learn_models, unit_spread)

cache_p = Path("data/phase11/footprint_dev.pkl")
if cache_p.exists():
    evs = pickle.load(open(cache_p, "rb"))
else:
    evs = collect(H.manifest_windows("dev"), per_window=None); pickle.dump(evs, open(cache_p, "wb"))
print("events", len(evs), {s: sum(e["session"] == s for e in evs) for s in "ABC"}, flush=True)
rng = np.random.default_rng(0)
res = {"selection_rule": "matched to detector-v2 (11A), E3-aligned, SNR>=6", "n_events": len(evs)}
st = footprint_stats(evs)
res["footprint_units"] = st
res["footprint_summary"] = {k: float(np.median([r[k] for r in st if r[k] is not None])) for k in
                            ("n_contacts_significant", "n_positive_dominant", "axial_asymmetry_log_ratio", "extent_trough_um",
                             "extent_repol_um", "extent_pre_um", "width_ms", "line20_rel_residual")}
res["footprint_summary"]["n_units"] = len(st)
res["footprint_by_session"] = {s: {k: float(np.median([r[k] for r in st if r["session"] == s and r[k] is not None]))
                                   for k in ("extent_trough_um", "extent_repol_um", "axial_asymmetry_log_ratio", "line20_rel_residual")}
                               for s in sorted({r["session"] for r in st})}
print(json.dumps(res["footprint_summary"]), flush=True)

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--models", default="line20,linefree,radial,aniso,twoline"); _ap.add_argument("--out", default="reports/phase11/exp11b_dev.json")
_a = _ap.parse_args()
names = _a.models.split(",")
loso = {n: {"heldout_err": [], "insample": [], "spread": {}, "drop_disp": [], "ms": []} for n in names}
learned = {}
for test in ("A", "B", "C"):
    tr = [e for e in evs if e["session"] != test]; te = [e for e in evs if e["session"] == test]
    sub = [tr[i] for i in rng.choice(len(tr), min(1500, len(tr)), replace=False)]
    mb, md = learn_models(sub, cache={})
    learned[test] = {"radial": mb.radial.tolist()}
    allm = {"line20": F.Model("line20"), "linefree": F.Model("linefree"), "radial": mb, "aniso": md,
            "twoline": F.Model("twoline", n_src=2), "line20c": F.Model("line20c", n_src=2)}
    models = {n: allm[n] for n in names}
    te_s = [te[i] for i in sorted(rng.choice(len(te), min(800, len(te)), replace=False))]
    for n, m in models.items():
        c = {}
        loso[n]["heldout_err"] += heldout_channel_error(m, te_s, cache=c).tolist()
        t = time.perf_counter(); fits = [F.fit(m, e["a"], e["sigma"], e["E"], cache=c) for e in te_s]
        loso[n]["ms"].append((time.perf_counter() - t) * 1e3 / len(te_s))
        loso[n]["insample"] += [f.sse_rel for f in fits]
        # primary source for twoline = larger amplitude component
        prim = []
        for f in fits:
            if n == "twoline":
                k = int(np.argmax(f.amps)); f = F.Fit(f.theta[2 * k:2 * k + 2], f.amps, f.sse, f.sse_rel)
            prim.append(f)
        loso[n]["spread"].update(unit_spread(te_s, prim))
        dr = np.random.default_rng(9)
        for e, f in zip(te_s[:300], prim[:300]):
            keep = np.ones(len(e["a"]), bool); keep[dr.choice(len(keep), 8, replace=False)] = False
            g = F.fit(m, e["a"], e["sigma"], e["E"], mask=keep)
            if n == "twoline":
                k = int(np.argmax(g.amps)); th = g.theta[2 * k:2 * k + 2]
            else:
                th = g.theta[:2]
            loso[n]["drop_disp"].append(float(np.linalg.norm(th - f.theta[:2])))
    print("LOSO", test, "done", flush=True)

def boot_rel(a, b, n=2000, seed=1):
    a, b = np.asarray(a), np.asarray(b); r = np.random.default_rng(seed); i = r.integers(0, len(a), (n, len(a)))
    v = np.median(b[i], axis=1) / np.median(a[i], axis=1) - 1
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

base = loso["line20"]
res["loso"] = {}
for n in names:
    x = loso[n]; units = sorted(set(base["spread"]) & set(x["spread"]))
    sp_a = [base["spread"][u] for u in units]; sp_b = [x["spread"][u] for u in units]
    res["loso"][n] = {"heldout_contact_rel_err_median": float(np.median(x["heldout_err"])),
                      "heldout_vs_line20": [float(np.median(x["heldout_err"]) / np.median(base["heldout_err"]) - 1), boot_rel(base["heldout_err"], x["heldout_err"])],
                      "insample_rel_sse_median": float(np.median(x["insample"])),
                      "unit_spread_median_um": float(np.median(sp_b)), "n_units": len(units),
                      "spread_vs_line20": [float(np.median(sp_b) / np.median(sp_a) - 1), boot_rel(sp_a, sp_b)],
                      "dropout8_disp_median_um": float(np.median(x["drop_disp"])), "dropout8_disp_p90_um": float(np.percentile(x["drop_disp"], 90)),
                      "fit_ms": float(np.mean(x["ms"]))}
    print(n, json.dumps(res["loso"][n]), flush=True)
res["learned_radial_by_heldout_session"] = learned
Path(_a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
