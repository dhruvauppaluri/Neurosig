"""Evaluate a factory on real + null events (null = same units, event times shifted by +20 ms)."""
from __future__ import annotations

import numpy as np

from edge.phase9 import harness as H
from edge.phase9.events import sorter_snippets
from scripts.phase5.data import load_window, window_path


def build_events_shift(wins, shift_s=0.0, conditioning="raw", max_events=60):
    out = []
    for w in wins:
        p = window_path(w["pid"], w["t0_s"], w["duration_s"])
        if not p.exists():
            continue
        d = load_window(p)
        v = H.condition_window(d, conditioning)
        d = dict(d); d["sorter_times_s"] = d["sorter_times_s"] + shift_s
        arr, sn, mt = sorter_snippets(d, "probe", max_events=max_events, seed=int(w["t0_s"]), voltages=v)
        out.append({"window": w, "array": arr, "snippets": sn, "meta": mt})
    return out


def evaluate(factory, real, null, dropout=True, noise=True):
    r = H.run(real, factory, dropout=dropout, noise=noise)
    n = H.run(null, factory, dropout=False, noise=False)
    for m in r:
        r[m]["null_spread_um"] = n[m]["within_unit_spread_um"]["median"]
        r[m]["null_rank_corr"] = n[m]["unit_depth_rank_corr_median"]
        sp = r[m]["within_unit_spread_um"]["median"]
        r[m]["real_over_null_spread"] = (sp / r[m]["null_spread_um"]) if r[m]["null_spread_um"] else None
    return r


def table(res, cols=("drop_clustered_0.75", "dropU_random_0.5", "dropU_clustered_0.75", "noise_independent",
                     "noise_correlated", "noise_impulsive", "noise_gain", "noise_drift"), methods=None):
    methods = methods or list(next(iter(res.values())))
    lines = []
    for m in methods:
        lines.append(f"== {m}")
        lines.append("variant".ljust(20) + " valid axMed spread null  r/n   rho  " + " ".join(c.replace("noise_", "n_").replace("dropU_", "U_").replace("drop_", "D_")[:9].ljust(9) for c in cols))
        for v, r in res.items():
            x = r[m]
            row = f"{v.ljust(20)} {x['valid_fraction']:.2f} {x['axial_disagreement_um']['median']:5.1f} {x['within_unit_spread_um']['median']:6.1f} {x['null_spread_um']:5.1f} {x['real_over_null_spread']:.2f} {x['unit_depth_rank_corr_median']:5.2f}  "
            row += " ".join((f"{x[c]['median']:5.1f}    " if c in x and x[c].get('n') else "na       ") for c in cols)
            lines.append(row)
    return "\n".join(lines)
