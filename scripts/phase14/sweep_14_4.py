"""Phase 14.4 development sweep of online-association parameters (dev sessions A-C only). One registry entry per
configuration is written by log_registry14.py from the saved JSON. Usage: sweep_14_4.py"""
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from scripts.phase14.run_14_4 import run_session
from edge.phase14.online_units import Params

ROUND = sys.argv[1] if len(sys.argv) > 1 else "1"
GRIDS = {"1": [dict(), dict(tau_assign=1.4), dict(tau_assign=1.4, margin=0.4), dict(min_gain=40), dict(min_gain=80),
               dict(min_gain=80, tau_assign=1.4), dict(min_gain=80, radius=40.0), dict(min_gain=80, margin_abs=20),
               dict(min_gain=120, margin_abs=20), dict(min_gain=80, margin_abs=20, n_init=12, snr_new=8.0)],
         "2": [dict(min_gain=80, merge_every=10 ** 9), dict(min_gain=80, a_lo=0.8, a_hi=1.25),
               dict(min_gain=80, a_lo=0.8, a_hi=1.25, merge_every=10 ** 9), dict(min_gain=80, a_lo=0.8, a_hi=1.25, tau_new=2.4),
               dict(min_gain=80, a_lo=0.8, a_hi=1.25, tau_new=2.4, merge_every=10 ** 9),
               dict(min_gain=80, a_lo=0.8, a_hi=1.25, tau_new=2.4, margin=0.4)]}
GRIDS["3"] = [dict(min_gain=80, merge_every=10 ** 9, snr_assign=5.0), dict(min_gain=80, merge_every=10 ** 9, snr_assign=6.0),
              dict(min_gain=60, merge_every=10 ** 9, snr_assign=6.0), dict(min_gain=80, merge_every=10 ** 9, snr_assign=6.0, margin=0.4)]
GRIDS["4"] = [dict(min_gain=80, merge_every=10 ** 9, snr_assign=6.0, relink_quiet_s=3.0),
              dict(min_gain=80, merge_every=10 ** 9, snr_assign=6.0, relink_quiet_s=3.0, relink_tau=2.4)]
GRID = GRIDS[ROUND]
out = []
for g in GRID:
    p = Params(**g); res = {}
    for s in "ABC":
        e = run_session(s, p)[0]
        res[s] = {k: e[k] for k in ("precision", "coverage_all", "coverage_hq", "precision_hq", "merge_rate", "fragmentation_mean", "ari", "spread_reduction", "ms_per_event") if k in e}
    m = {k: float(np.median([res[s].get(k, np.nan) for s in res])) for k in res["A"]}
    mn = {k: float(np.min([res[s].get(k, np.nan) for s in res])) for k in ("precision", "coverage_hq")}
    out.append({"params": g, "median": m, "min": mn, "sessions": res})
    print(g, {k: round(v, 3) for k, v in m.items()}, "min prec", round(mn["precision"], 3), flush=True)
Path(f"reports/phase14/exp14_4_dev_sweep{ROUND}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
