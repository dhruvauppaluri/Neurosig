"""G-series: combine retained conditioners; robust_scan v2 overlap rule. Dev windows (real+null) and dev synthetic."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from edge.localize import get_localizer
from edge.localize.beamform import default_scan_grid_um
from edge.phase9 import harness as H
from edge.phase9.evalset import build_events_shift, evaluate, table
from edge.phase9.factories import baseline_factory
from edge.phase9.improved import RobustScanV2, conditioned_factory
S="/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/"
wins = H.manifest_windows("dev")
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
C = dict(baseline=True, car=True, interp_dead=True)

def v2_factory(min_snr):
    def f(array):
        e = dict(baseline_factory(array))
        shift = np.array([0.0, float(array.xyz_um()[:, 1].min()), 0.0])
        e["robust_scan"] = RobustScanV2(array=array, grid_xyz_um=default_scan_grid_um(array.n_channels) + shift,
                                         model="line_source", min_snr=min_snr)
        return e
    return f
def cond(f): return conditioned_factory(f, **C)
variants = {"B": conditioned_factory(baseline_factory, baseline=True), "C=B+car+interp": cond(baseline_factory),
            "C+robust_v2(snr4)": cond(v2_factory(4.0)), "C+robust_v2(snr6)": cond(v2_factory(6.0))}
res = {k: evaluate(f, real, null) for k, f in variants.items()}
json.dump(res, open(S+"gseries.json","w"), default=float)
print(table(res))
