"""Calibrate (dev only) and evaluate the P14 event pipeline.

  eval_p14.py calibrate <tag>      fit thr3, the p_overlap logistic, the region level q and the K=1 position choice on
                                   data/phase14/p14rows_p14dev<tag>.pkl; writes reports/phase14/p14_calibration<tag>.json
  eval_p14.py evaluate <pool> <tag>  score a pool with the dev calibration (thr2 is the pool's own label-free
                                   self-calibration); writes reports/phase14/p14_eval_<pool><tag>.json
Complete populations: every generated event is scored; no event is dropped. Primary source = highest-SNR source.
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase10 import overlap_stress as O
from edge.phase14.count_eval import wilson
from edge.phase14.fast14 import FastSourceModel
from edge.phase14.pipeline14 import P14Config, covered, p_overlap, region

mode = sys.argv[1]


def load(pool, tag):
    return pickle.load(open(f"data/phase14/p14rows_{pool}{tag}.pkl", "rb"))


def grid_for(r):
    E = np.asarray(r["E"]) if "E" in r else None
    return E


def primary(r):
    return np.asarray(r["sources"])[int(np.argmax(r["snr"]))]


def grid_of(E):
    m = FastSourceModel.__new__(FastSourceModel)
    ylo, yhi = E[:, 1].min() - 40, E[:, 1].max() + 40
    xs = np.arange(-10.0, 90.0 + 1e-9, 10.0); ys = np.arange(ylo, yhi + 1e-9, 10.0)
    return np.array([(x, y) for y in ys for x in xs]), np.array([-10.0, ylo]), np.array([90.0, ys[-1]])


def attach_E(pool, rows):
    """Regenerate the contact geometry of each event (deterministic generator, same seed) for region scoring."""
    from edge.phase14 import bench14 as B
    ev = B.generate(pool, 150, 60, 60)
    for r, e in zip(rows, ev):
        assert r["kind"] == e.kind and np.allclose(r["sources"], e.sources)
        r["E"] = np.asarray(e.snippet.electrode_xyz_um)[:, :2]
    return rows


def sigma_hat(r, level: float = 6.0):
    """Identifiability scale: sqrt(area / pi) of the coarse-grid profile region at a fixed level (>= one cell)."""
    a = max(int(np.sum(r["region_delta"] <= level)), 1) * 100.0
    return float(np.sqrt(a / np.pi))


def q_needed(r):
    P, lo, hi = grid_of(r["E"])
    tr = primary(r)
    i = int(np.argmin(np.sum((P - tr[None]) ** 2, axis=1)))
    if not np.all(np.abs(P[i] - tr) <= 5.0 + 1e-6):
        return np.inf
    return float(r["region_delta"][i])


if mode == "calibrate":
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    d = load("p14dev", tag); rows = attach_E("p14dev", d["rows"])
    pairs = [r for r in rows if r["kind"] == "pair" and r["n"] >= 2 and np.isfinite(r["F32"])]
    thr3 = float(np.quantile([r["F32"] for r in pairs], 0.95)) if pairs else np.inf
    from scipy.optimize import minimize
    S = np.array([np.log(max(r["F21"], 1e-9)) for r in rows if r["kind"] == "single"])
    Pp = np.array([np.log(max(r["F21"], 1e-9)) for r in rows if r["kind"] in ("pair", "triple")])
    x = np.r_[S, Pp]; y = np.r_[np.zeros(len(S)), np.ones(len(Pp))]; w = np.r_[np.full(len(S), 0.5 / len(S)), np.full(len(Pp), 0.5 / len(Pp))]
    nll = lambda b: float(np.sum(w * (np.logaddexp(0, b[0] + b[1] * x) - y * (b[0] + b[1] * x))))
    lr_w = minimize(nll, np.zeros(2)).x.tolist()
    qs = np.sort([q_needed(r) for r in rows])
    k = int(np.ceil(0.9 * (len(qs) + 1))) - 1
    q = float(qs[min(k, len(qs) - 1)])
    sing = [r for r in rows if r["kind"] == "single"]
    use_ms3 = np.median([np.linalg.norm(r["p1_ms3"] - r["sources"][0]) for r in sing]) <= np.median([np.linalg.norm(r["p1"] - r["sources"][0]) for r in sing])
    dist = np.array([np.linalg.norm((r["p1_ms3"] if use_ms3 else r["p1"]) - primary(r)) for r in rows])
    ratio = np.sort(dist / np.array([sigma_hat(r) for r in rows]))
    k2 = int(np.ceil(0.9 * (len(ratio) + 1))) - 1
    q_disc = float(ratio[min(k2, len(ratio) - 1)])
    r_fixed = float(np.sort(dist)[min(k2, len(dist) - 1)])
    e_fl = float(np.median([np.linalg.norm(r["p1"] - r["sources"][0]) for r in sing]))
    e_ms = float(np.median([np.linalg.norm(r["p1_ms3"] - r["sources"][0]) for r in sing]))
    cal = {"thr3": thr3, "lr_w": lr_w, "q_region": q, "q_disc": q_disc, "r_fixed": r_fixed, "q_region_finite": bool(np.isfinite(q)),
           "k1_position": "fastline20" if e_fl <= e_ms else "ms3", "dev_single_err_fastline20": e_fl, "dev_single_err_ms3": e_ms,
           "n_dev_events": len(rows), "frac_truth_outside_grid": float(np.mean(~np.isfinite(qs)))}
    Path(f"reports/phase14/p14_calibration{tag}.json").write_text(json.dumps(cal, indent=1) + "\n")
    print(cal)

else:
    pool, tag = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    cal = json.loads(Path(f"reports/phase14/p14_calibration{tag}.json").read_text()) if Path(f"reports/phase14/p14_calibration{tag}.json").exists() \
        else json.loads(Path("reports/phase14/frozen_p14.json").read_text())["p14_calibration"]
    d = load(pool, tag); rows = attach_E(pool, d["rows"])
    cfg = P14Config(lr_w=tuple(cal["lr_w"]), thr3=cal["thr3"])
    kind = np.array([r["kind"] for r in rows]); res = np.array([r["res"] for r in rows]); cond = np.array([r["cond"] for r in rows])
    n = np.array([1 if r["n"] == 1 else (3 if (np.isfinite(r["F32"]) and r["F32"] > cal["thr3"]) else 2) for r in rows])
    for r in rows:                                  # K=1 position: the dev-selected estimator
        r["p1_sel"] = r["p1_ms3"] if cal["k1_position"] == "ms3" else r["p1"]
    out = {"pool": pool, "tag": tag, "thr2_selfcal": d["thr2"], "calibration": cal}
    sing = kind == "single"
    k = int((n[sing] >= 2).sum()); N = int(sing.sum())
    out["false_two"] = {"rate": k / N, "ci": wilson(k, N), "n": N, "by_cond": {c: float(np.mean(n[sing & (cond == c)] >= 2)) for c in sorted(set(cond[sing]))}}
    for nm, sel in (("pairs_all", kind == "pair"), ("pairs_resolvable", (kind == "pair") & res), ("pairs_simultaneous_res", (kind == "pair") & res & (cond == "dt0"))):
        kk = int((n[sel] >= 2).sum()); NN = int(sel.sum()); out[nm] = {"recall": kk / max(NN, 1), "ci": wilson(kk, NN), "n": NN}
    out["recall_by_cond"] = {c: {"all": float(np.mean(n[(kind == "pair") & (cond == c)] >= 2)),
                                 "res": float(np.mean(n[(kind == "pair") & (cond == c) & res] >= 2)) if ((kind == "pair") & (cond == c) & res).any() else None}
                             for c in sorted(set(cond[kind == "pair"]))}
    errs = []; both = 0
    for r, nn in zip(rows, n):
        if r["kind"] == "pair" and r["res"] and nn >= 2:
            m = O.match(np.asarray(r["sources"]), np.asarray(r["p2"]))
            if m["matched"] == 2:
                both += 1; errs += m["errs"]
    out["pairs_res_both_matched"] = both / max(int(((kind == "pair") & res).sum()), 1)
    out["pair_err_median_um"] = float(np.median(errs)) if errs else None
    tri = kind == "triple"; ok3 = 0
    for r, nn in zip([r for r in rows if r["kind"] == "triple"], n[tri]):
        if nn == 3 and O.match(np.asarray(r["sources"]), np.asarray(r["p3"]))["matched"] == 3:
            ok3 += 1
    out["triples"] = {"declared3": float(np.mean(n[tri] == 3)), "recovered3": ok3 / max(int(tri.sum()), 1), "declared_ge2": float(np.mean(n[tri] >= 2)),
                      "false3_on_pairs": float(np.mean(n[kind == "pair"] == 3)), "false3_on_singles": float(np.mean(n[sing] == 3))}
    out["single_err_median_um"] = float(np.median([np.linalg.norm(r["p1_sel"] - r["sources"][0]) for r in rows if r["kind"] == "single"]))
    out["single_err_p90_um"] = float(np.percentile([np.linalg.norm(r["p1_sel"] - r["sources"][0]) for r in rows if r["kind"] == "single"], 90))
    out["single_err_median_um_fastline20"] = float(np.median([np.linalg.norm(r["p1"] - r["sources"][0]) for r in rows if r["kind"] == "single"]))
    # calibrated overlap probability (deployment prior) and confident errors
    po = np.array([p_overlap(r["F21"], cfg, 0.5) for r in rows])           # balanced prior on the balanced bench
    ytrue = (kind != "single").astype(float)
    bins = np.linspace(0, 1, 11); rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        mm = (po >= a) & (po < b) if b < 1 else (po >= a)
        if mm.any():
            rel.append({"bin": [float(a), float(b)], "n": int(mm.sum()), "mean_p": float(po[mm].mean()), "frac_multi": float(ytrue[mm].mean())})
    out["p_overlap_reliability_balanced_prior"] = rel
    out["p_overlap_ece"] = float(sum(x["n"] * abs(x["mean_p"] - x["frac_multi"]) for x in rel) / len(po))
    conf = np.maximum(po, 1 - po) >= 0.9
    wrong = ((po >= 0.5) != (ytrue > 0))
    out["confident_count_decisions"] = {"frac_confident": float(conf.mean()), "confidently_wrong_frac_of_all": float((conf & wrong).mean()),
                                        "confidently_wrong_frac_of_confident": float((conf & wrong).sum() / max(conf.sum(), 1))}
    dec_wrong = ((n >= 2) != (ytrue > 0))
    out["hard_decision_count_error"] = {"all": float(dec_wrong.mean()), "singles": float(dec_wrong[sing].mean()), "multi": float(dec_wrong[~sing].mean())}
    # complete-population primary-source regions
    q = cal["q_region"]; cov, area, unb = [], [], []
    for r in rows:
        P, lo, hi = grid_of(r["E"])
        m, a, u = region(r["region_delta"], q, P, lo, hi)
        cov.append(covered(r["region_delta"], q, P, primary(r))); area.append(a); unb.append(u)
    cov, area, unb = np.array(cov), np.array(area), np.array(unb)
    snr_p = np.array([max(r["snr"]) for r in rows])
    sig_hat = np.array([sigma_hat(r) for r in rows])
    dist = np.array([np.linalg.norm(r["p1_sel"] - primary(r)) for r in rows])
    disc_cov = dist <= cal["q_disc"] * sig_hat; disc_area = np.pi * (cal["q_disc"] * sig_hat) ** 2
    fix_cov = dist <= cal["r_fixed"]
    out["discs"] = {"normalized": {"coverage_all": float(disc_cov.mean()), "coverage_ci": wilson(int(disc_cov.sum()), len(disc_cov)),
                                   "median_area_um2": float(np.median(disc_area)), "unbounded_frac": float(unb.mean()),
                                   "by_kind": {kk: float(disc_cov[kind == kk].mean()) for kk in ("single", "pair", "triple")},
                                   "by_snr": {f"{a}-{b}": float(disc_cov[(snr_p >= a) & (snr_p < b)].mean()) for a, b in ((0, 6), (6, 10), (10, 15), (15, 99)) if ((snr_p >= a) & (snr_p < b)).any()}},
                    "fixed": {"coverage_all": float(fix_cov.mean()), "radius_um": cal["r_fixed"], "area_um2": float(np.pi * cal["r_fixed"] ** 2),
                              "by_snr": {f"{a}-{b}": float(fix_cov[(snr_p >= a) & (snr_p < b)].mean()) for a, b in ((0, 6), (6, 10), (10, 15), (15, 99)) if ((snr_p >= a) & (snr_p < b)).any()}}}
    out["regions"] = {"coverage_all": float(cov.mean()), "coverage_ci": wilson(int(cov.sum()), len(cov)), "n": int(len(cov)),
                      "median_area_um2": float(np.median(area)), "unbounded_frac": float(unb.mean()),
                      "coverage_bounded": float(cov[~unb].mean()) if (~unb).any() else None,
                      "median_area_bounded_um2": float(np.median(area[~unb])) if (~unb).any() else None,
                      "by_kind": {kk: {"coverage": float(cov[kind == kk].mean()), "unbounded": float(unb[kind == kk].mean()),
                                       "median_area": float(np.median(area[kind == kk]))} for kk in ("single", "pair", "triple")},
                      "by_snr": {f"{a}-{b}": {"coverage": float(cov[(snr_p >= a) & (snr_p < b)].mean()), "n": int(((snr_p >= a) & (snr_p < b)).sum()),
                                             "unbounded": float(unb[(snr_p >= a) & (snr_p < b)].mean())}
                                 for a, b in ((0, 6), (6, 10), (10, 15), (15, 99)) if ((snr_p >= a) & (snr_p < b)).any()}}
    # runtime
    tc = np.array([r["ms_count"] for r in rows]); tp = np.array([r["ms_position"] for r in rows])
    out["runtime_ms"] = {"count_median": float(np.median(tc)), "count_p95": float(np.percentile(tc, 95)), "count_p99": float(np.percentile(tc, 99)),
                         "position_median": float(np.median(tp)), "total_median": float(np.median(tc + tp)), "total_p95": float(np.percentile(tc + tp, 95))}
    Path(f"reports/phase14/p14_eval_{pool}{tag}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(pool, tag, "FT", round(out["false_two"]["rate"], 3), out["false_two"]["ci"], "res", round(out["pairs_resolvable"]["recall"], 3),
          "sim", round(out["pairs_simultaneous_res"]["recall"], 3), "perr", out["pair_err_median_um"], "tri", out["triples"]["recovered3"],
          "serr", round(out["single_err_median_um"], 1), "cov", round(out["regions"]["coverage_all"], 3), "unb", round(out["regions"]["unbounded_frac"], 3),
          "area", out["regions"]["median_area_um2"], "disc cov", round(out["discs"]["normalized"]["coverage_all"], 3), "disc area", round(out["discs"]["normalized"]["median_area_um2"]),
          "fixed cov", round(out["discs"]["fixed"]["coverage_all"], 3), "fixed area", round(out["discs"]["fixed"]["area_um2"]), "ECE", round(out["p_overlap_ece"], 3), "confwrong", round(out["confident_count_decisions"]["confidently_wrong_frac_of_all"], 3),
          "ms", round(out["runtime_ms"]["total_median"], 2))
