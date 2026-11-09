"""Phase 12.5b real-data proxy coverage (no ground truth). Uses the 12.3 per-event rows (mean3 line20 estimates and
the line20 identifiability area of every matched event).

Reference for an event: the median position of the OTHER events of its unit (leave-one-out, >= 9 others), a unit-level
proxy with its own small uncertainty and possible drift. Region: a disc of area c * identifiability_area (the area is
an approximate 68% profile-likelihood region), c fitted on the dev sessions so 90% of events cover their reference,
optionally per SNR group (Mondrian). Evaluated on the select sessions D-P. Reported against a fixed-radius disc fitted
the same way (no per-event information), so any gain is attributable to the identifiability measure.
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase11.calibration import snr_group, wilson

V = "mean3"


def refs(rows):
    by = defaultdict(list)
    for r in rows:
        if r["xy"][V] is not None:
            by[(r["session"], r["unit"])].append(r)
    out = []
    for u, L in by.items():
        if len(L) < 10:
            continue
        P = np.array([r["xy"][V] for r in L])
        for i, r in enumerate(L):
            ref = np.median(np.delete(P, i, axis=0), axis=0)
            out.append({"d": float(np.linalg.norm(P[i] - ref)), "area": max(r["ident_area"], 64.0), "g": snr_group(r["snr"]), "session": r["session"]})
    return out


dev = refs(pickle.load(open("data/phase12/rows12_3_dev.pkl", "rb")))
EVAL = sys.argv[1] if len(sys.argv) > 1 else "select"
sel = refs(pickle.load(open(f"data/phase12/rows12_3_{EVAL}.pkl", "rb")))
# disc radius = sqrt(c * area / pi); covered iff d <= radius  <=>  pi d^2 / area <= c
score = lambda r: np.pi * r["d"] ** 2 / r["area"]


def cq(v, level=0.9):
    v = np.sort(v); n = len(v); return float(v[min(n - 1, int(np.ceil((n + 1) * level)) - 1)])


c_all = cq([score(r) for r in dev])
c_g = {g: cq([score(r) for r in dev if r["g"] == g]) for g in ("low", "mid", "high")}
r_fix = cq([r["d"] for r in dev])
out = {"reference": "leave-one-out unit median of mean3 estimates (>= 9 other events)", "c_pooled": c_all, "c_by_snr": c_g,
       "fixed_radius_um": r_fix}
for name, rows in (("dev", dev), (EVAL, sel)):
    o = {}
    for rule, cov, area in (("ident_pooled", lambda r: score(r) <= c_all, lambda r: c_all * r["area"]),
                            ("ident_mondrian_snr", lambda r: score(r) <= c_g[r["g"]], lambda r: c_g[r["g"]] * r["area"]),
                            ("fixed_disc", lambda r: r["d"] <= r_fix, lambda r: np.pi * r_fix ** 2)):
        k = sum(cov(r) for r in rows)
        o[rule] = {"coverage": k / len(rows), "ci95": list(wilson(k, len(rows))), "area_median_um2": float(np.median([area(r) for r in rows])),
                   "by_snr": {g: float(np.mean([cov(r) for r in rows if r["g"] == g])) for g in ("low", "mid", "high")},
                   "per_session_min": float(min(np.mean([cov(r) for r in rows if r["session"] == s]) for s in {r["session"] for r in rows}))}
    o["n"] = len(rows)
    out[name] = o
    print(name, {k: (round(v["coverage"], 3), round(v["area_median_um2"]), {g: round(x, 3) for g, x in v["by_snr"].items()}, round(v["per_session_min"], 3))
                 for k, v in o.items() if k != "n"}, o["n"])
Path(f"reports/phase12/exp12_5_real_proxy{'' if EVAL == 'select' else '_' + EVAL}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
