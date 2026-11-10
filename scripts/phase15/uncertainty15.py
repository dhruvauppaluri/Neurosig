"""Phase 15 Track A uncertainty: leave-one-morphology-out (LOMO) Mondrian conformal discs for the A-P15 position
estimator (mean of the ms3 and line20 centres, both spatially whitened) over all 13 BBP morphologies (Phase 14 dev,
select and final pools, NP1). No fresh morphology exists any more, so this is cross-validated evidence, not a new
single-use final. Mondrian bins: observed SNR tertile (dev edges) x declared overlap (F21 > dev 98th pct).
Writes reports/phase15/uncertainty15_lomo.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase14.count_eval import wilson

rows = []
for pool in ("p14dev", "p14select", "p14final"):
    for x in pickle.load(open(f"data/phase14/rows_{pool}.pkl", "rb")):
        i = int(np.argmax(x["snr"]))
        rows.append(dict(pool=pool, cell=x["cells"][i], kind=x["kind"], cond=x["cond"], snr=x["snr_obs"], F=x["ms3_sp"]["F21"],
                         d_ens=float(np.linalg.norm(0.5 * (x["ms3_sp"]["p1"] + x["line20_sp"]["p1"]) - np.asarray(x["sources"])[i])),
                         d_ms3=float(np.linalg.norm(x["ms3_sp"]["p1"] - np.asarray(x["sources"])[i])),
                         d_l20=float(np.linalg.norm(x["line20_sp"]["p1"] - np.asarray(x["sources"])[i]))))
dev = [r for r in rows if r["pool"] == "p14dev"]
edges = np.quantile([r["snr"] for r in dev], [1 / 3, 2 / 3]); Fthr = np.quantile([r["F"] for r in dev if r["kind"] == "single"], 0.98)
b = lambda r: int(np.digitize(r["snr"], edges)) * 2 + int(r["F"] > Fthr)


def q(x, a=0.9):
    x = np.sort(x); return float(x[min(int(np.ceil(a * (len(x) + 1))) - 1, len(x) - 1)])


out = {"estimators": {}, "snr_edges": edges.tolist()}
for est in ("d_ens", "d_l20", "d_ms3"):
    per = {}; allcov = []; allarea = []; cond_cov = {}
    for m in sorted({r["cell"] for r in rows}):
        tr = [r for r in rows if r["cell"] != m]; te = [r for r in rows if r["cell"] == m]
        R = {k: q([r[est] for r in tr if b(r) == k]) for k in set(b(r) for r in tr)}
        cov = [r[est] <= R[b(r)] for r in te]; area = [np.pi * R[b(r)] ** 2 for r in te]
        per[m] = {"coverage": float(np.mean(cov)), "n": len(te), "median_err": float(np.median([r[est] for r in te]))}
        allcov += cov; allarea += area
        for r, c in zip(te, cov):
            for key in (f"kind:{r['kind']}", f"snr_bin:{int(np.digitize(r['snr'], edges))}", f"cond:{r['cond']}"):
                cond_cov.setdefault(key, []).append(c)
    out["estimators"][est] = {"coverage_all": float(np.mean(allcov)), "ci": wilson(int(np.sum(allcov)), len(allcov)),
                              "median_area_um2": float(np.median(allarea)), "per_morphology": per,
                              "min_morphology_coverage": float(min(v["coverage"] for v in per.values())),
                              "by_condition": {k: float(np.mean(v)) for k, v in sorted(cond_cov.items())},
                              "median_err_all": float(np.median([r[est] for r in rows])),
                              "p90_err_all": float(np.percentile([r[est] for r in rows], 90))}
    e = out["estimators"][est]
    print(est, "cov", round(e["coverage_all"], 3), "area", round(e["median_area_um2"]), "min morph", round(e["min_morphology_coverage"], 3),
          "err", round(e["median_err_all"], 1), round(e["p90_err_all"], 1), {k: round(v, 2) for k, v in e["by_condition"].items() if k.startswith(("kind", "snr"))})
Path("reports/phase15/uncertainty15_lomo.json").write_text(json.dumps(out, indent=1) + "\n")
