"""Single-shot Phase 9 final evaluation of the frozen P9 configuration. Refuses to rerun without a documented protocol."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.evaluate import folded_error_um
from edge.phase8.registry import versions
from edge.phase8.synth import make_events as legacy_synth
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, P9_PARAMS, corrected_baseline_factory, legacy_factory, p9_factory
from edge.phase9.evalset import build_events_shift, evaluate
from edge.phase9.synth_real import make_events as real_synth, xy_error
from edge.phase9.improved import condition

OUT = Path("reports/phase9/final_locked.json")
if OUT.exists() and "--force-rerun-final" not in sys.argv:
    raise SystemExit(f"{OUT} exists; the locked sessions are evaluated once per frozen configuration.")
facs = {"legacy_raw": legacy_factory, "corrected_B": corrected_baseline_factory, "P9": p9_factory}
report = {"versions": versions(), "p9_hash": P9_HASH, "p9_params": P9_PARAMS, "sets": {}}
for kind, label in (("locked", "Phase 6 locked_final sessions E-H: evaluated once for P9"),
                    ("heldout5", "Phase 5 held-out windows: PREVIOUSLY USED benchmark (not a fresh holdout)")):
    wins = H.manifest_windows(kind)
    real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
    t = time.time()
    report["sets"][kind] = {"label": label, "n_windows": len(wins),
                            "results": {k: evaluate(f, real, null) for k, f in facs.items()}}
    print(kind, round(time.time() - t), "s", flush=True)
# synthetic final pools (known truth)
array, syn = real_synth("final", n_per_noise=100, contaminated_fraction=0.5)
rs = {}
for k, f in facs.items():
    rs[k] = {}
    for m, e in f(array).items():
        a = np.array([xy_error(r.xyz_um, ev.truth_xy) if (r := e.estimate(ev.snippet)).available else np.nan for ev in syn])
        c = np.array([ev.contaminated for ev in syn])
        def st(x):
            ok = np.isfinite(x)
            return {"accepted": float(ok.mean()), "median": float(np.median(x[ok])), "p90": float(np.percentile(x[ok], 90)),
                    "p95": float(np.percentile(x[ok], 95)), "large_gt50": float(np.mean(x[ok] > 50))}
        rs[k][m] = {"all": st(a), "single": st(a[~c]), "contaminated": st(a[c])}
report["synthetic_real_like_final"] = {"n": len(syn), "metric": "xy error um (probe plane)", "results": rs}
ev = legacy_synth("final", n_per_cell=60)
legacy_arr = ev[0].snippet
from edge.benchmark_v3 import _array
la = _array("linear")
ls = {}
for k, f in {"legacy": None, "P9": p9_factory}.items():
    if f is None:
        from edge.benchmark_v3 import _engines
        eng = _engines(la)
    else:
        eng = f(la)
    ls[k] = {}
    for m, e in eng.items():
        errs = []
        for x in ev:
            r = e.estimate(x.snippet)
            errs.append(folded_error_um(r.xyz_um, x.truth_xyz_um, x.snippet.electrode_xyz_um) if r.available else np.nan)
        a = np.array(errs); ok = np.isfinite(a)
        ls[k][m] = {"accepted": float(ok.mean()), "median": float(np.median(a[ok])), "p90": float(np.percentile(a[ok], 90)),
                    "p95": float(np.percentile(a[ok], 95)), "large_gt50": float(np.mean(a[ok] > 50))}
report["synthetic_legacy_broad_final"] = {"n": len(ev), "note": "Phase 8 'final' pool: lambda=80 um monopole footprint, 2.5 ms spikes, no DC/shared noise",
                                          "metric": "folded error um", "results": ls}
OUT.write_text(json.dumps(report, indent=1, default=float) + "\n")
print("written")
