"""Phase 14.4: sorter-free online unit association and unit-level localization on real IBL detector events.

Usage: .venv/bin/python scripts/phase14/run_14_4.py <split> [param=value ...] [--tag name]
  split: dev (A-C) | select (D-S) | final (T-W; only via run_final14.py) | a comma list of sessions
Sorter labels are read ONLY by the evaluation (edge/phase14/online_units.evaluate) after the stream is processed.
Writes reports/phase14/exp14_4_<split>[_tag].json.
"""
import json, pickle, sys, time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase12.fast import FastLine20
from edge.phase14 import detcache as DC
from edge.phase14.online_units import OnlineUnits, Params, evaluate

SPLITS = {"dev": list("ABC"), "select": list("DEFGHIJKLMNOPQRS"), "final": list("TUVW")}


def quality_map(session):
    for p in ("data/phase12/cluster_info.pkl", "data/phase14/cluster_info14.pkl"):
        if Path(p).exists():
            d = pickle.load(open(p, "rb"))
            if session in d:
                return dict(zip(d[session]["cluster_id"], d[session]["label"]))
    return None


def pairwise_median(P):
    if len(P) < 2:
        return np.nan
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    return float(np.median(d[np.triu_indices(len(P), 1)]))


def unit_localization(c, labels, fl, sigma, N=10, max_units=200, seed=0):
    """Per online unit with >= 2N members: event-level spread (median pairwise distance of single-event FastLine20
    positions, up to 40 events) and unit-level spread (disjoint N-event pools, mean amplitude vector, FastLine20)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in np.unique(labels[labels >= 0])[:max_units]:
        idx = np.flatnonzero(labels == u)
        if len(idx) < 2 * N:
            continue
        idx = rng.permutation(idx)
        singles = np.array([x for x in (fl.fit(c["amp3"][i], sigma, iters=5)[0] for i in idx[:40]) if x is not None])
        pools = [idx[k * N:(k + 1) * N] for k in range(min(len(idx) // N, 8))]
        pooled = np.array([x for x in (fl.fit(c["amp3"][p].mean(axis=0), sigma, iters=5)[0] for p in pools) if x is not None])
        if len(singles) < 2 or len(pooled) < 2:
            continue
        rows.append({"unit": int(u), "n": int(len(idx)), "event_spread": pairwise_median(singles),
                     "unit_spread": pairwise_median(pooled), "pos": pooled.mean(axis=0).tolist()})
    return rows


def run_session(s, params: Params):
    p = Path(f"data/phase14/det_{s}.npz")
    if not p.exists():
        return None
    c = DC.load(p)
    ou = OnlineUnits(c["E"], params)
    t = time.perf_counter()
    for i in range(len(c["t"])):
        ou.add(c["wave"][i].astype(float), c["amp3"][i], int(c["ch"][i]), float(c["t"][i]), float(c["snr"][i]), i)
    ms = (time.perf_counter() - t) * 1e3 / max(len(c["t"]), 1)
    lab = ou.final_labels()
    qm = quality_map(s)
    qual = np.array([qm.get(int(u), np.nan) if (qm and u >= 0) else np.nan for u in c["sorter_unit"]]) if qm else None
    ev = evaluate(lab, c["sorter_unit"], c["snr"], None if qual is None else np.nan_to_num(qual, nan=-1))
    ev["ms_per_event"] = ms
    ev["n_units_created"] = len(ou.units); ev["n_active"] = int(sum(u.active and u.alive for u in ou.units))
    fl = FastLine20(np.c_[c["E"], np.zeros(len(c["E"]))])
    sigma = np.median(c["sigma_rc"], axis=0).astype(float)
    loc = unit_localization(c, lab, fl, sigma)
    ev["loc_units"] = len(loc)
    if loc:
        es = np.array([r["event_spread"] for r in loc]); us = np.array([r["unit_spread"] for r in loc])
        ev["event_spread_median"] = float(np.nanmedian(es)); ev["unit_spread_N10_median"] = float(np.nanmedian(us))
        ev["spread_reduction"] = 1 - ev["unit_spread_N10_median"] / ev["event_spread_median"]
    # initialization time: first event of each sorter unit (>= 20 matched events) -> first event assigned to an
    # active online unit whose majority sorter label is that unit
    maj = {}
    for u in np.unique(lab[(lab >= 0) & (c["sorter_unit"] >= 0)]):
        v, n = np.unique(c["sorter_unit"][(lab == u) & (c["sorter_unit"] >= 0)], return_counts=True)
        maj[int(u)] = int(v[np.argmax(n)])
    init = []
    for su in np.unique(c["sorter_unit"][c["sorter_unit"] >= 0]):
        k = np.flatnonzero(c["sorter_unit"] == su)
        if len(k) < 20:
            continue
        ok = [i for i in k if lab[i] >= 0 and maj.get(int(lab[i])) == su]
        init.append((float(c["t"][ok[0]] - c["t"][k[0]]) if ok else np.inf, int(np.searchsorted(k, ok[0]) if ok else len(k))))
    if init:
        ev["init_events_median"] = float(np.median([x[1] for x in init]))
        ev["init_never_frac"] = float(np.mean([np.isinf(x[0]) for x in init]))
    ev["memory_kib_units"] = float(sum(u.mu.nbytes + u.amp.nbytes for u in ou.units if u.alive) / 1024)
    return ev, lab, loc


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "")
    split = args[0]
    sessions = SPLITS.get(split, split.split(","))
    kw = {}
    for a in args[1:]:
        k, v = a.split("="); kw[k] = type(getattr(Params(), k))(float(v))
    params = Params(**kw)
    out = {"versions": versions(), "split": split, "params": asdict(params), "sessions": {}}
    for s in sessions:
        r = run_session(s, params)
        if r is None:
            out["sessions"][s] = {"status": "missing_dataset"}; continue
        out["sessions"][s] = r[0]
        e = r[0]
        print(s, f"prec {e['precision']:.3f} cov {e['coverage_all']:.3f} covHQ {e['coverage_hq']:.3f} precHQ {e['precision_hq']:.3f} "
              f"merge {e['merge_rate']:.2f} frag {e['fragmentation_mean']:.2f} ari {e['ari']:.2f} units {e['n_active']} "
              f"red {e.get('spread_reduction', np.nan):.2f} ms {e['ms_per_event']:.2f}", flush=True)
    ok = [v for v in out["sessions"].values() if "precision" in v]
    keys = ("precision", "coverage_all", "coverage_matched", "coverage_hq", "precision_hq", "merge_rate", "fragmentation_mean",
            "duplicate_units_frac", "ari", "spread_reduction", "ms_per_event")
    out["summary"] = {k: {"median": float(np.nanmedian([v.get(k, np.nan) for v in ok])),
                          "min": float(np.nanmin([v.get(k, np.nan) for v in ok]))} for k in keys} if ok else {}
    name = f"exp14_4_{split}{'_' + tag if tag else ''}.json"
    Path("reports/phase14", name).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("summary", {k: round(v["median"], 3) for k, v in out["summary"].items()})


if __name__ == "__main__":
    main()
