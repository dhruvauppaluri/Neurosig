"""Phase 14.2-14.3: source-count decision rules on cached statistics (data/phase14/rows_<pool>.pkl) and their scoring.

Every rule's threshold is calibrated on p14dev singles only (target false-two rate, default 2%), then applied unchanged
to other pools. Scoring is over the complete population of each family (no event is dropped): singles (false-two
rate, all conditions), all pairs, resolvable pairs, per-condition pairs and triples. Pair positions are matched to the
true somata by optimal assignment with a 60 um gate (edge/phase10/overlap_stress.match).
"""

from __future__ import annotations

import numpy as np

from edge.phase10 import overlap_stress as O
from edge.phase12.count_model import LogisticCount

SPECS = ("line20", "ms", "ms5")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def features(r: dict, suf: str = "") -> np.ndarray:
    """Learned-rule features (all computed from the snippet; no truth). ``suf`` selects the noise variant."""
    f = []
    for s in SPECS:
        x = r[s + suf]
        f += [np.log1p(max(x["F21"], 0.0)), np.log(max(x["chi1"], 1e-6))]
    m = r["ms" + suf]
    f += [m["c1_pol"], np.log1p(m["c1_trough"]), np.log(max(m["en_ratio"], 1e-6)), min(m["sep"], 200) / 60.0,
          min(m["dt_comp"], 30) / 10.0, np.log1p(max(m.get("F32", 0.0), 0.0)),
          r.get("rank2", 0.0), min(r.get("n_trough", 0), 4), min(r.get("t_spread", 0.0), 10.0), np.log1p(r["snr_obs"]),
          np.log(max(r.get("lr_prob", 1e-6), 1e-6) / max(1 - r.get("lr_prob", 1e-6), 1e-6))]
    return np.array(f, float)


FEATURE_NAMES = [f"{k}_{s}" for s in SPECS for k in ("logF21", "logchi1")] + [
    "ms_c1_pol", "ms_log_c1_trough", "ms_log_en_ratio", "ms_sep_60", "ms_dtcomp_10", "ms_logF32", "rank2", "n_trough",
    "t_spread", "log_snr", "lr_logit"]


def score_value(r: dict, rule: str, model=None) -> float:
    """Continuous score; 'two' is declared when score > threshold."""
    if rule == "lr":
        return r.get("lr_prob", -np.inf)
    if rule.endswith("_F"):
        return r[rule[:-2]]["F21"]
    if rule.endswith("_Ftr"):
        x = r[rule[:-4]]
        ok = (x["c1_pol"] >= 0.5) and (x["c1_trough"] >= 4.0)
        return x["F21"] if ok else -np.inf
    if rule.startswith("learn"):
        return float(model.prob(features(r, rule[5:]))[0])
    raise ValueError(rule)


def calibrate(rows: list[dict], rule: str, target: float = 0.02, model=None) -> float:
    s = np.array([score_value(r, rule, model) for r in rows if r["kind"] == "single"])
    return float(np.quantile(s[np.isfinite(s)] if np.isfinite(s).any() else s, 1 - target)) if np.isfinite(s).sum() > 10 else float(np.quantile(s, 1 - target))


def positions(r: dict, rule: str, two: bool) -> np.ndarray:
    spec = ("ms" + rule[5:]) if rule.startswith("learn") else (rule.rsplit("_", 1)[0] if rule != "lr" else None)
    if rule == "lr":
        return np.asarray(r.get("lr_pos", np.zeros((1, 2))))
    x = r[spec]
    return x["p2"] if two else x["p1"][None]


def evaluate(rows: list[dict], rule: str, thr: float, model=None, thr3: float | None = None) -> dict:
    out = {}
    sc = np.array([score_value(r, rule, model) for r in rows])
    two = sc > thr
    kind = np.array([r["kind"] for r in rows]); res = np.array([r["res"] for r in rows]); cond = np.array([r["cond"] for r in rows])
    if rule == "lr":
        two = np.array([r.get("lr_n", 1) == 2 for r in rows]) if thr is None else two
    sing = kind == "single"
    k = int(two[sing].sum()); n = int(sing.sum())
    out["false_two"] = {"rate": k / max(n, 1), "ci": wilson(k, n), "n": n}
    out["false_two_by_cond"] = {c: float(two[sing & (cond == c)].mean()) for c in sorted(set(cond[sing]))}
    out["single_accept"] = 1 - k / max(n, 1)
    for name, sel in (("pairs_all", kind == "pair"), ("pairs_resolvable", (kind == "pair") & res),
                      ("pairs_simultaneous_res", (kind == "pair") & res & (cond == "dt0"))):
        kk = int(two[sel].sum()); nn = int(sel.sum())
        out[name] = {"recall": kk / max(nn, 1), "ci": wilson(kk, nn), "n": nn}
    out["recall_by_cond"] = {c: {"all": float(two[(kind == "pair") & (cond == c)].mean()),
                                 "res": float(two[(kind == "pair") & (cond == c) & res].mean()) if ((kind == "pair") & (cond == c) & res).any() else None}
                             for c in sorted(set(cond[kind == "pair"]))}
    # position error on correctly recovered (both sources matched within 60 um) resolvable pairs
    errs, both = [], 0
    for r, t in zip(rows, two):
        if r["kind"] == "pair" and r["res"] and t:
            m = O.match(np.asarray(r["sources"]), positions(r, rule, True))
            if m["matched"] == 2:
                both += 1; errs += list(m["errs"])
    nres = int(((kind == "pair") & res).sum())
    out["pairs_res_both_matched"] = both / max(nres, 1)
    out["pair_err_median_um"] = float(np.median(errs)) if errs else np.nan
    # singles: primary position error of the K=1 answer (accepted singles)
    e1 = [float(np.linalg.norm(positions(r, rule, False)[0] - r["sources"][0])) for r, t in zip(rows, two) if r["kind"] == "single" and not t]
    out["single_err_median_um"] = float(np.median(e1)) if e1 else np.nan
    # ranking quality singles vs resolvable pairs
    s0 = sc[sing]; s1 = sc[(kind == "pair") & res]
    s0 = np.where(np.isfinite(s0), s0, -1e9); s1 = np.where(np.isfinite(s1), s1, -1e9)
    out["auc_single_vs_res"] = float(np.mean(s1[:, None] > s0[None]) + 0.5 * np.mean(s1[:, None] == s0[None]))
    if thr3 is not None:
        tr = kind == "triple"
        sp = positions.__defaults__ if False else None
        key = ("ms" + rule[5:]) if rule.startswith("learn") else (rule.rsplit("_", 1)[0] if rule != "lr" else "ms")
        f32 = np.array([r[key].get("F32", 0.0) for r in rows])
        three = two & (f32 > thr3)
        ok = 0
        for r, t in zip([r for r in rows if r["kind"] == "triple"], three[tr]):
            if t and O.match(np.asarray(r["sources"]), r[key]["p3"])["matched"] == 3:
                ok += 1
        out["triples"] = {"declared3": float(three[tr].mean()), "recovered3": ok / max(int(tr.sum()), 1), "n": int(tr.sum()),
                          "false3_on_pairs": float(three[kind == "pair"].mean()), "false3_on_singles": float(three[sing].mean())}
    return out


def pr_curve(rows: list[dict], rule: str, model=None, n: int = 25) -> list[dict]:
    """Recall (resolvable pairs, all pairs) vs false-two rate (singles) over thresholds."""
    sc = np.array([score_value(r, rule, model) for r in rows])
    kind = np.array([r["kind"] for r in rows]); res = np.array([r["res"] for r in rows])
    s0 = sc[kind == "single"]; fin = s0[np.isfinite(s0)]
    out = []
    for q in np.linspace(0.5, 0.999, n):
        t = float(np.quantile(fin, q)) if len(fin) else np.inf
        out.append({"false_two": float(np.mean(s0 > t)), "recall_res": float(np.mean(sc[(kind == "pair") & res] > t)),
                    "recall_all": float(np.mean(sc[kind == "pair"] > t)),
                    "precision_res": float(np.sum(sc[(kind == "pair") & res] > t) / max(np.sum(sc[(kind == "pair") & res] > t) + np.sum(s0 > t), 1))})
    return out


def train_learned(rows: list[dict], target: float = 0.02, suf: str = ""):
    X = np.array([features(r, suf) for r in rows if r["kind"] in ("single", "pair")])
    y = np.array([r["kind"] == "pair" for r in rows if r["kind"] in ("single", "pair")], int)
    Xs = np.array([features(r, suf) for r in rows if r["kind"] == "single"])
    m = LogisticCount().fit(X, y, Xs, target_fp=target)
    return m
