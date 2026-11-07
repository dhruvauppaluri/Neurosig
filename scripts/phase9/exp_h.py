"""H: synthetic check (known truth, dev pools) of conditioners and robust v2 rule."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from edge.benchmark_v3 import _array, _engines
from edge.localize.beamform import default_scan_grid_um
from edge.phase8.evaluate import folded_error_um
from edge.phase8.overlap_bench import make_overlap_events
from edge.phase8.synth import make_events
from edge.phase9.improved import RobustScanV2, condition, noise_aware_overlap
from edge.localize.overlap import spatial_overlap_conflict

array = _array("linear")
eng = _engines(array)
eng["robust_v2"] = RobustScanV2(array=array, min_snr=4.0)
ev = make_events("dev", n_per_cell=60)
variants = {"raw": {}, "C": dict(baseline=True, car=True, interp_dead=True)}
print("accuracy on dev synthetic (n=%d; folded error um; accepted only; large>50)" % len(ev))
for vn, opt in variants.items():
    for name, e in eng.items():
        errs = []
        for x in ev:
            sn = condition(x.snippet, **opt) if opt else x.snippet
            r = e.estimate(sn)
            errs.append(folded_error_um(r.xyz_um, x.truth_xyz_um, sn.electrode_xyz_um) if r.available else np.nan)
        a = np.array(errs); ok = np.isfinite(a)
        print(f"  {vn:4s} {name:13s} accepted={ok.mean():.2f} med={np.median(a[ok]):5.1f} p90={np.percentile(a[ok],90):5.1f} large={np.mean(a[ok]>50):.3f}")
oe = make_overlap_events("dev", n_per_noise=30)
print("overlap rule on dev overlap bench: rejection rate by kind/noise")
for rule, f in (("old", lambda s: spatial_overlap_conflict(s)[0]), ("v2_snr4", lambda s: noise_aware_overlap(s, min_snr=4.0)[0]),
                ("v2_snr6", lambda s: noise_aware_overlap(s, min_snr=6.0)[0])):
    out = {}
    for kind in ("single", "pair", "triple"):
        for lab, sel in (("all", lambda e: True), ("resolvable", lambda e: e.resolvable)):
            es = [e for e in oe if e.kind == kind and sel(e)]
            if es: out[f"{kind}/{lab}"] = round(float(np.mean([f(e.snippet) for e in es])), 3)
    print("  ", rule, out)
