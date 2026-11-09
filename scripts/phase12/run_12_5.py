"""Phase 12.5: complete-population uncertainty evaluation. Usage: run_12_5.py

Fixes the Phase 11 scoring gap: Phase 11 scored the primary-source region only on events whose primary had a matched
truth. Here EVERY event of a pool is in the denominator.

Per event (P12-LR count decision, frozen count model):
  primary_covered   the primary region (nominal 90%) contains the truth source it is matched to (Hungarian, 60 um);
                    an unmatched primary (phantom / count error) or an event with no estimate counts as NOT covered.
  sources_covered   fraction of true sources inside some predicted region (missed sources count as uncovered).
  extra_regions     predicted sources with no truth (phantom regions), reported separately.
Count/identity uncertainty is scored separately from position: the logistic P(two sources) is evaluated with the
Brier score, a 10-bin reliability table and the false-confidence rate (P >= 0.9 or <= 0.1 but wrong).

Regions: Gauss-Newton covariance of the declared solution (scaled by max(chi2_red, 1)), region = {d^2 <= q}.
Calibrations compared (fitted on p12dev, evaluated on p12select):
  gauss       q = chi2(2) 90% quantile (4.61), no conformal step
  P11-mond    Phase 11 Mondrian over stratum x SNR group x dropout, matched primaries only (the old protocol)
  P12-full    Mondrian over declared count x SNR group x chi1 tercile x dropout, with the conformal quantile taken
              over ALL events of the group (unmatched primaries have score +inf), conservative pooling below n = 30
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase10.overlap_stress import frozen_calibration, match
from edge.phase11.calibration import snr_group, wilson
from edge.phase12 import count_model as CM

CAL = frozen_calibration()
cm = json.loads(Path("reports/phase12/count_model_p12.json").read_text())
LR = CM.LogisticCount.from_json(cm["logistic"])
GAUSS_Q = 4.605


def load(pool):
    d = pickle.load(open(f"data/phase12/bench_{pool}.pkl", "rb"))
    return [(e, r) for e, r in zip(d["events"], d["recs"])]


def rows_for(pool, chi_edges=None):
    out = []
    for ev, a in load(pool):
        if a["none"]:
            out.append({"ev": ev, "none": True}); continue
        d = CM.decide_lr(a, CAL, LR)
        cov = d.cov
        sc = max(a["chi1"], 1.0)
        m = match(ev.sources, d.pos)
        # per predicted source: d^2 to its matched truth (inf if unmatched)
        from scipy.optimize import linear_sum_assignment
        D = np.linalg.norm(ev.sources[:, None] - d.pos[None], axis=2)
        ri, ci = linear_sum_assignment(D)
        d2 = np.full(len(d.pos), np.inf); tr_of = {}
        for i, j in zip(ri, ci):
            if D[i, j] <= 60.0:
                C = cov[2 * j:2 * j + 2, 2 * j:2 * j + 2]
                C = 0.5 * (C + C.T) + 1e-6 * np.eye(2)
                if np.all(np.isfinite(C)):
                    e = d.pos[j] - ev.sources[i]; d2[j] = float(e @ np.linalg.solve(C, e)); tr_of[j] = i
        dets = [float(np.linalg.det(0.5 * (cov[2 * j:2 * j + 2, 2 * j:2 * j + 2] + cov[2 * j:2 * j + 2, 2 * j:2 * j + 2].T) + 1e-6 * np.eye(2)))
                for j in range(len(d.pos))]
        out.append({"ev": ev, "none": False, "n": d.n, "d2": d2, "det": dets, "snr_g": snr_group(a["snr_obs"]), "chi1": a["chi1"],
                    "dead": "dead" if a["n_dead"] else "none", "p2": float(LR.prob(a["x"])[0]), "flags": d.flags,
                    "stratum": ("shared" if "shared_waveform_pair" in d.flags else ("two" if d.n >= 2 else ("flagged" if d.flags else "clean"))),
                    "n_true": len(ev.sources), "matched": m["matched"]})
    return out


def chi_tercile(rows, edges):
    for r in rows:
        if not r["none"]:
            r["chi_t"] = int(np.searchsorted(edges, r["chi1"]))


def fit_q(rows, keyf, all_events: bool, min_n=30, level=0.9):
    groups = defaultdict(list)
    for r in rows:
        if r["none"]:
            continue
        s = r["d2"][0]
        if not all_events and not np.isfinite(s):
            continue
        groups[keyf(r)].append(s)
    def cq(v):            # split-conformal order statistic; +inf scores (unmatched primaries) sort last
        v = np.sort(np.asarray(v, float)); n = len(v)
        return float(v[min(n - 1, int(np.ceil((n + 1) * level)) - 1)]) if n else np.inf
    pooled = [s for v in groups.values() for s in v]
    qall = cq(pooled)
    table = {}
    for k, v in groups.items():
        table[k] = cq(v) if len(v) >= min_n else max(qall, cq(v))
    return table, qall


def evaluate(rows, qf):
    cov_primary, src_cov, src_n, extra, areas, by = [], 0, 0, 0, [], defaultdict(list)
    for r in rows:
        if r["none"]:
            cov_primary.append(False); src_n += len(r["ev"].sources); continue
        q = qf(r)
        c = bool(np.isfinite(r["d2"][0]) and r["d2"][0] <= q)
        cov_primary.append(c)
        covered = sum(1 for j in range(r["n"]) if np.isfinite(r["d2"][j]) and r["d2"][j] <= q)
        src_cov += covered; src_n += r["n_true"]; extra += int(np.sum(~np.isfinite(r["d2"])))
        areas.append(np.pi * q * np.sqrt(max(r["det"][0], 0)) if np.isfinite(q) else np.inf)
        by[(r["ev"].family, r["ev"].kind, r["snr_g"])].append(c)
    k = int(np.sum(cov_primary))
    return {"n_events": len(rows), "primary_coverage_all_events": k / len(rows), "ci95": list(wilson(k, len(rows))),
            "true_source_coverage": src_cov / max(src_n, 1), "phantom_regions": extra,
            "area_median_um2": float(np.median(areas)) if areas else None,
            "unbounded_region_fraction": float(np.mean(~np.isfinite(areas))) if areas else None,
            "by_family_kind_snr": {"|".join(map(str, kk)): {"n": len(v), "coverage": float(np.mean(v))} for kk, v in sorted(by.items())}}


def count_calibration(rows):
    R = [r for r in rows if not r["none"]]
    y = np.array([r["n_true"] >= 2 for r in R], float); p = np.array([r["p2"] for r in R])
    bins = np.minimum((p * 10).astype(int), 9)
    rel = [{"bin": b / 10, "n": int(np.sum(bins == b)), "mean_p": float(p[bins == b].mean()), "frac_true_multi": float(y[bins == b].mean())}
           for b in range(10) if np.any(bins == b)]
    conf_wrong = float(np.mean(((p >= 0.9) & (y == 0)) | ((p <= 0.1) & (y == 1))))
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": rel, "false_confidence_rate": conf_wrong,
            "base_rate_multi": float(y.mean())}


EVAL = sys.argv[1] if len(sys.argv) > 1 else "p12select"
dev, sel = rows_for("p12dev"), rows_for(EVAL)
edges = np.quantile([r["chi1"] for r in dev if not r["none"]], [1 / 3, 2 / 3])
chi_tercile(dev, edges); chi_tercile(sel, edges)
k11 = lambda r: (r["stratum"], r["snr_g"], r["dead"])
k12 = lambda r: (min(r["n"], 2), r["snr_g"], r["chi_t"], r["dead"])
t11, q11 = fit_q(dev, k11, all_events=False)
t12, q12 = fit_q(dev, k12, all_events=True)
qf = {"gauss": lambda r: GAUSS_Q, "P11-mond": lambda r: t11.get(k11(r), q11), "P12-full": lambda r: t12.get(k12(r), q12)}
out = {"versions": versions(), "count_rule": "P12-LR (reports/phase12/count_model_p12.json)", "chi1_tercile_edges": edges.tolist(),
       "P12_table": {"|".join(map(str, k)): v for k, v in t12.items()}, "dev": {}, "select": {}}
for name, f in qf.items():
    for split, rows in (("dev", dev), ("select", sel)):
        out[split][name] = evaluate(rows, f)
        o = out[split][name]
        print(f"{split:6s} {name:9s} primary cov (all events) {o['primary_coverage_all_events']:.3f} {np.round(o['ci95'], 3)} "
              f"true-source cov {o['true_source_coverage']:.3f} phantom {o['phantom_regions']} area {o['area_median_um2']} unbounded {o['unbounded_region_fraction']}")
for split, rows in (("dev", dev), ("select", sel)):
    out[split]["count_uncertainty"] = count_calibration(rows)
    print(split, "count:", {k: v for k, v in out[split]["count_uncertainty"].items() if k != "reliability"})
# risk-coverage for the select pool: accept the most confident fraction by region area, report primary coverage & error
R = [r for r in sel if not r["none"]]
order = np.argsort([r["det"][0] for r in R])
rc = []
for frac in (0.5, 0.7, 0.8, 0.9, 1.0):
    S = [R[i] for i in order[:int(frac * len(R))]]
    rc.append({"accept_fraction": frac, "primary_matched": float(np.mean([np.isfinite(r["d2"][0]) for r in S])),
               "coverage_P12": float(np.mean([np.isfinite(r["d2"][0]) and r["d2"][0] <= qf["P12-full"](r) for r in S]))})
out["select"]["risk_coverage"] = rc
out["select"]["P12-full"]["by_snr"] = {g: float(np.mean([np.isfinite(r["d2"][0]) and r["d2"][0] <= qf["P12-full"](r) for r in sel if not r["none"] and r["snr_g"] == g]))
                                        for g in ("low", "mid", "high")}
Path(f"reports/phase12/exp12_5_uncertainty{'' if EVAL == 'p12select' else '_' + EVAL}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
(None if EVAL != "p12select" else Path("reports/phase12/calibration_p12.json")) and Path("reports/phase12/calibration_p12.json").write_text(json.dumps({"keys": "declared count (1|2) x snr_group x chi1 tercile x dropout",
    "chi1_tercile_edges": edges.tolist(), "table": out["P12_table"], "pooled": q12, "fitted_on": "p12dev, all events"}, indent=1) + "\n")
print(json.dumps(rc), out["select"]["P12-full"]["by_snr"])
