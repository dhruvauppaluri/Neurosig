"""Phase 12.2: unit-level pooling stability. Usage: run_12_2.py dev|select [--population matched|all]

dev     = Phase 5 calibration+selection windows of sessions A-C (development)
select  = sessions D (Phase 5 held-out), E-H (Phase 6), I-L (Phase 10), M-P (Phase 11) - all previously used, read for
          model selection / generalisation checks only. Phase 5 held-out windows of A-C are excluded so that select is
          session-disjoint from dev.
Population 'matched' = sorter events with a detector-v2.0 partner (the Phase 11 matched-event population);
'all' = every sorter event (includes events without a measurable trough). Units need >= 10 events (N = 5 needs 2 pools).
Writes reports/phase12/exp12_2_<split>_<population>.json and data/phase12/rows12_2_<split>_<population>.pkl.
"""
import argparse, json, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9 import harness as H
from edge.phase9.events import window_array
from edge.phase11.configs import P11_HASH, p11_factory
from edge.phase12 import pooling as PL
from edge.phase12.splits import split_windows
from scripts.phase5.data import load_window, window_path



def session_events(sess, ws):
    p = Path(f"data/phase12/sess_{sess}.npz")
    w0 = next(w for w in ws if window_path(w["pid"], w["t0_s"], w["duration_s"]).exists())
    array = window_array(load_window(window_path(w0["pid"], w0["t0_s"], w0["duration_s"])))
    if p.exists():
        return PL.load_session(p, array)
    se = PL.collect_session(ws, sess, log=lambda s: print(s, flush=True))
    PL.save_session(se, p)
    return se


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--population", default="matched")
    ap.add_argument("--sessions", default=""); a = ap.parse_args()
    t0 = time.time()
    wins = split_windows(a.split)
    by = defaultdict(list)
    for w in wins:
        by[w["session"]].append(w)
    only = set(a.sessions.split(",")) if a.sessions else None
    rows, positions, pop_stats = [], {}, {}
    for sess, ws in sorted(by.items()):
        if only and sess not in only:
            continue
        se = session_events(sess, ws)
        ul = PL.UnitLocalizer(se, p11_factory(se.array)["weighted_scan"])
        keep = se.matched if a.population == "matched" else np.ones(len(se.unit), bool)
        units = {}
        for u in np.unique(se.unit):
            idx = np.flatnonzero((se.unit == u) & keep)
            if len(idx) >= 10:
                units[int(u)] = idx
        pop_stats[sess] = {"events_stored": int(len(se.unit)), "events_before_cap": se.extra["n_total_before_cap"],
                           "matched_fraction": float(se.matched.mean()), "snr_lt4_fraction": float(np.mean(se.snr < 4)),
                           "matched_snr_lt4_fraction": float(np.mean(se.snr[se.matched] < 4)) if se.matched.any() else None,
                           "units_total": int(len(np.unique(se.unit))), "units_used": len(units),
                           "events_in_used_units": int(sum(len(v) for v in units.values()))}
        r = PL.unit_stability(ul, units, seed=1)
        rows += r
        for u, idx in units.items():
            positions[f"{sess}:{u}"] = {"n": int(len(idx)), "sorter_ch_y": float(np.median(se.E[se.sorter_ch[idx], 1])),
                                        "sorter_ch_x": float(np.median(se.E[se.sorter_ch[idx], 0])),
                                        "snr_median": float(np.median(se.snr[idx])), **PL.unit_position(ul, idx[:200])}
        print(sess, pop_stats[sess], f"{time.time() - t0:.0f}s", flush=True)
    keys = sorted({k for r in rows for k in r if (k.startswith(("ws_", "l20_")) and not k.endswith(("first_last", "valid")))})
    res = {"versions": versions(), "split": a.split, "population": a.population, "engine_config": f"P11 {P11_HASH}",
           "ns": list(PL.NS), "max_pools": PL.MAX_POOLS, "population_stats": pop_stats,
           "summary": PL.summarize_stability(rows, keys),
           "per_session": {s: PL.summarize_stability([r for r in rows if r["session"] == s], ["ws_single", "ws_mean", "l20_single", "l20_joint", "l20_mean"])
                           for s in sorted({r["session"] for r in rows})},
           "runtime_s": time.time() - t0}
    tag = f"{a.split}_{a.population}" + (f"_{a.sessions.replace(',', '')}" if a.sessions else "")
    Path(f"reports/phase12/exp12_2_{tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    pickle.dump({"rows": rows, "positions": positions}, open(f"data/phase12/rows12_2_{tag}.pkl", "wb"))
    S = res["summary"]
    for N in S:
        print(N, {k: (round(v["median"], 1), None if not v["vs_single"] else round(v["vs_single"]["rel_change"], 3))
                  for k, v in S[N].items() if isinstance(v, dict)})
