"""Phase 11F conditional uncertainty calibration (Mondrian split conformal on squared Mahalanobis distance of the
primary source). Groups: stratum (clean / flagged / two / shared) x SNR group (low <8, mid 8-16, high >=16, peak
contact trough / MAD noise, estimated from the data) x dropout (none / dead contacts present). A group with fewer than
MIN_N calibration events uses the larger of its own quantile (if any) and its stratum-level quantile (conservative
pooling); this is disclosed per group. Calibration on p11dev, evaluation on p11select / p11final.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_N = 30
SNR_EDGES = (8.0, 16.0)


def snr_group(snr: float) -> str:
    return "low" if snr < SNR_EDGES[0] else ("mid" if snr < SNR_EDGES[1] else "high")


def stratum(flags, n_src) -> str:
    if "shared_waveform_pair" in flags:
        return "shared"
    if n_src >= 2:
        return "two"
    return "flagged" if flags else "clean"


def conformal_q(d2, level=0.9):
    d2 = np.sort(np.asarray(d2, float))
    if not len(d2):
        return None
    k = min(int(np.ceil((len(d2) + 1) * level)), len(d2)) - 1
    return float(d2[k])


def fit_groups(rows, level=0.9, keys=("stratum", "snr_g", "drop")):
    """rows: dicts with d2, stratum, snr_g, drop (accepted events only)."""
    by = defaultdict(list); by_s = defaultdict(list)
    for r in rows:
        by[tuple(r[k] for k in keys)].append(r["d2"]); by_s[r["stratum"]].append(r["d2"])
    qs = {s: conformal_q(v, level) for s, v in by_s.items()}
    qall = conformal_q([r["d2"] for r in rows], level)
    table, info = {}, {}
    for g, v in by.items():
        own = conformal_q(v, level)
        pooled = qs.get(g[0], qall)
        if len(v) >= MIN_N:
            table[g] = own; info[g] = ("own", len(v))
        else:
            table[g] = max(own or 0.0, pooled); info[g] = ("pooled_conservative", len(v))
    return {"table": table, "info": info, "stratum": qs, "all": qall, "keys": keys}


def q_for(model, r):
    g = tuple(r[k] for k in model["keys"])
    if g in model["table"]:
        return model["table"][g]
    return model["stratum"].get(r["stratum"], model["all"])


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def evaluate(model, rows, large_um=50.0, conf_hi=0.8):
    cov, area = [], []
    for r in rows:
        q = q_for(model, r)
        cov.append(r["d2"] <= q)
        area.append(float(np.pi * q * np.sqrt(max(r["det"], 0.0))))
    cov = np.array(cov); area = np.array(area)
    out = {"n": len(rows), "coverage": float(cov.mean()), "coverage_ci95": wilson(int(cov.sum()), len(cov)),
           "area_median_um2": float(np.median(area))}
    for key in ("snr_g", "stratum", "drop"):
        out[f"by_{key}"] = {}
        for g in sorted({r[key] for r in rows}):
            m = np.array([r[key] == g for r in rows])
            out[f"by_{key}"][g] = {"n": int(m.sum()), "coverage": float(cov[m].mean()), "ci95": wilson(int(cov[m].sum()), int(m.sum())),
                                   "area_median_um2": float(np.median(area[m]))}
    # false confidence: region area small (most confident third) yet error > large_um
    thr = np.percentile(area, 33)
    err = np.array([r["err"] for r in rows])
    out["false_confidence_rate"] = float(np.mean((area <= thr) & (err > large_um)))
    order = np.argsort(area)
    out["risk_coverage"] = [{"kept": f, "median_err": float(np.median(err[order[:max(int(f * len(err)), 1)]])),
                             "large_rate": float(np.mean(err[order[:max(int(f * len(err)), 1)]] > large_um))}
                            for f in (0.5, 0.7, 0.9, 1.0)]
    return out
