"""Null control: are within-unit spreads informative, or do static offsets/anchoring make any event 'stable'?"""
import sys, copy
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.events import sorter_snippets
from edge.phase9.factories import baseline_factory
from edge.phase9.improved import conditioned_factory
from scripts.phase5.data import load_window, window_path

wins = H.manifest_windows("dev")
def build(shift_s):
    out = []
    for w in wins:
        d = load_window(window_path(w["pid"], w["t0_s"], w["duration_s"]))
        d = dict(d); d["sorter_times_s"] = d["sorter_times_s"] + shift_s
        arr, sn, mt = sorter_snippets(d, "probe", max_events=60, seed=int(w["t0_s"]))
        out.append({"window": w, "array": arr, "snippets": sn, "meta": mt})
    return out
real, null = build(0.0), build(0.020)
for name, f in {"baseline": baseline_factory, "E2_baseline_sub": conditioned_factory(baseline_factory, baseline=True),
                "E2+E3_sub+car": conditioned_factory(baseline_factory, baseline=True, car=True)}.items():
    r, n = H.run(real, f, dropout=False, noise=False), H.run(null, f, dropout=False, noise=False)
    print("==", name)
    for m in ("amplitude", "beamform", "weighted_scan", "template"):
        print(f"  {m:13s} real spread={r[m]['within_unit_spread_um']['median']:5.1f} rho={r[m]['unit_depth_rank_corr_median']:.2f} w/b={r[m]['within_over_between_unit_spread']:.2f} | "
              f"null spread={n[m]['within_unit_spread_um']['median']:5.1f} rho={n[m]['unit_depth_rank_corr_median']:.2f} w/b={n[m]['within_over_between_unit_spread']:.2f}")
