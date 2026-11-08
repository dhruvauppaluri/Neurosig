"""Phase 11B model selection on the selection split (previously used windows; not a fresh holdout): within-unit
spread of matched events (11A rule, E3 alignment, NO SNR filter), per session, for line20 / linefree / radial /
line20c, plus 8-contact dropout displacement. Empirical radial is learned from dev events only."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase11 import forward_models as F
from edge.phase11.footprint import collect, learn_models, unit_spread

cp = Path("data/phase11/footprint_select_all.pkl")
if cp.exists():
    evs = pickle.load(open(cp, "rb"))
else:
    wins = H.manifest_windows("heldout5") + H.manifest_windows("locked") + H.manifest_windows("locked10")
    evs = collect(wins, per_window=150, keep_low=True); pickle.dump(evs, open(cp, "wb"))
dev = pickle.load(open("data/phase11/footprint_dev.pkl", "rb"))
rng = np.random.default_rng(3)
mb, _ = learn_models([dev[i] for i in rng.choice(len(dev), min(2000, len(dev)), replace=False)], cache={})
models = {"line20": F.Model("line20"), "linefree": F.Model("linefree"), "radial": mb, "line20c": F.Model("line20c", n_src=2)}
fits = {}
for n, m in models.items():
    c = {}
    fits[n] = [F.fit(m, e["a"], e["sigma"], e["E"], cache=c) for e in evs]
    print(n, "fitted", flush=True)
out = {"n_events": len(evs), "split": "Phase 5 held-out + E-H + I-L (previously used; model selection only)", "models": {}}
snr = np.array([e["snr"] for e in evs])
for n in models:
    sp = unit_spread(evs, fits[n]); base = unit_spread(evs, fits["line20"])
    units = sorted(set(sp) & set(base)); a = np.array([base[u] for u in units]); b = np.array([sp[u] for u in units])
    r = np.random.default_rng(1); i = r.integers(0, len(units), (2000, len(units)))
    ci = np.percentile(np.median(b[i], axis=1) / np.median(a[i], axis=1) - 1, [2.5, 97.5])
    dev_ = []
    by = {}
    for e, f in zip(evs, fits[n]):
        by.setdefault((e["session"], e["unit"]), []).append(f.theta[:2])
    for e, f in zip(evs, fits[n]):
        v = by[(e["session"], e["unit"])]
        if len(v) >= 5:
            dev_.append(float(np.linalg.norm(f.theta[:2] - np.median(np.array(v), axis=0))))
    per = {s: float(np.median([sp[u] for u in sp if u[0] == s])) for s in sorted({u[0] for u in sp})}
    disp = []
    dr = np.random.default_rng(9)
    for e, f in list(zip(evs, fits[n]))[::10]:
        keep = np.ones(len(e["a"]), bool); keep[dr.choice(len(keep), 8, replace=False)] = False
        g = F.fit(models[n], e["a"], e["sigma"], e["E"], mask=keep)
        disp.append(float(np.linalg.norm(g.theta[:2] - f.theta[:2])))
    out["models"][n] = {"unit_spread_median_um": float(np.median(b)), "n_units": len(units),
                        "rel_vs_line20": float(np.median(b) / np.median(a) - 1), "ci95": [float(ci[0]), float(ci[1])],
                        "dev_p90_um": float(np.percentile(dev_, 90)), "per_session_spread_um": per,
                        "dropout8_disp_median_um": float(np.median(disp)), "dropout8_disp_p90_um": float(np.percentile(disp, 90))}
    print(n, json.dumps(out["models"][n]), flush=True)
Path("reports/phase11/exp11b_select.json").write_text(json.dumps(out, indent=1) + "\n")
