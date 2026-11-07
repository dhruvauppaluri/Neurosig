"""E1: snippet conditioning/alignment across all five existing methods on development windows (A-D)."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.factories import baseline_factory

wins = H.manifest_windows("dev")
variants = {"raw": dict(conditioning="raw"), "filtered": dict(conditioning="filtered"),
            "filtered+align45": dict(conditioning="filtered", align_half=45),
            "raw+align45": dict(conditioning="raw", align_half=45)}
out = {}
for name, kw in variants.items():
    t = time.time()
    ev = H.build_events(wins, **kw)
    out[name] = H.run(ev, baseline_factory, dropout=False, noise=False)
    print(name, f"{time.time()-t:.0f}s", flush=True)
    for m, r in out[name].items():
        a = r["axial_disagreement_um"]
        print(f"   {m:13s} valid={r['valid_fraction']:.3f} axial med/p90={a['median']:.1f}/{a['p90']:.1f} spread={r['within_unit_spread_um']['median']:.1f} (n={r['within_unit_spread_um']['n_units']}) rho={r['unit_depth_rank_corr_median']:.2f} w/b={r['within_over_between_unit_spread']:.2f}")
json.dump(out, open("/private/tmp/claude-501/-Users-dhruvauppaluri--claude-spares-spare-zfwvDI/ab536295-60fa-4914-8ddc-e384323aba86/scratchpad/e1.json", "w"), default=float)
