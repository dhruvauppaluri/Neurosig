"""Phase 15 Track B: learned source-count discriminators evaluated leave-one-morphology-out over all 13 BBP models.

Features (all from the snippet, no truth): log F21 / log chi1 for every spatially whitened basis family (line20, ms,
ms3, ms5, msq, msn, msw), K=2 component descriptors of ms3 (polarity, trough, energy ratio, separation, component
time difference), log F32, temporal features (rank2, n_trough, t_spread), log SNR. Models: the frozen single statistic
(ms3_sp F21), L2 logistic regression, gradient-boosted trees (HistGradientBoosting, depth <= 3). For each held-out
morphology the threshold is set at 2% false two-source on the TRAINING morphologies' singles (deployable: no held-out
labels). Exploratory: the Phase 14 final morphologies have been used once already. Run with data/ibl-venv.
Writes reports/phase15/overlap_lomo.json."""
import json, pickle, sys  # noqa
from pathlib import Path
import numpy as np

import os
FAM = tuple(os.environ.get("FAM", "line20,ms,ms3,ms5,msq,msn,msw").split(","))


def feats(x):
    f = []
    for s in FAM:
        y = x[s + "_sp"]; f += [np.log1p(max(y["F21"], 0)), np.log(max(y["chi1"], 1e-6))]
    m = x["ms3_sp"]
    f += [m["c1_pol"], np.log1p(m["c1_trough"]), np.log(max(m["en_ratio"], 1e-6)), min(m["sep"], 200) / 60, min(m["dt_comp"], 30) / 10,
          np.log1p(max(m.get("F32", 0), 0)), x.get("rank2", 0.0), min(x.get("n_trough", 0), 4), min(x.get("t_spread", 0.0), 10.0), np.log1p(x["snr_obs"])]
    return np.array(f, float)


rows = []
for pool in ("p14dev", "p14select", "p14final"):
    for x in pickle.load(open(f"data/phase14/rows_{pool}.pkl", "rb")):
        rows.append(dict(pool=pool, cell=x["cells"][0], kind=x["kind"], res=x["res"], cond=x["cond"], X=feats(x), F=x["ms3_sp"]["F21"]))
X = np.array([r["X"] for r in rows]); X = np.nan_to_num(X, nan=0.0, posinf=50, neginf=-50)
kind = np.array([r["kind"] for r in rows]); res = np.array([r["res"] for r in rows]); cell = np.array([r["cell"] for r in rows])
cond = np.array([r["cond"] for r in rows]); F = np.array([r["F"] for r in rows])
y = (kind != "single").astype(int)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
QS = np.array([0.98, 0.99, 0.995, 0.998, 0.999])


def scorer(model, tr_mask, te_mask):
    if model == "F21":
        return F[tr_mask], F[te_mask]
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-9
    A, B = (X[tr_mask] - mu) / sd, (X[te_mask] - mu) / sd
    use = kind[tr_mask] != "triple"
    clf = (LogisticRegression(C=1.0, max_iter=3000) if model == "lr" else
           HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0))
    clf.fit(A[use], y[tr_mask][use])
    return clf.predict_proba(A)[:, 1], clf.predict_proba(B)[:, 1]


def nested_q(model, outer):
    """Training-quantile level whose inner leave-one-morphology-out false-two rate is <= 2% (mean over inner folds)."""
    fps = []
    for m in sorted(set(cell[outer])):
        tr = outer & (cell != m); te = outer & (cell == m)
        s_tr, s_te = scorer(model, tr, te)
        si = s_te[kind[te] == "single"]
        fps.append([np.mean(si > np.quantile(s_tr[kind[tr] == "single"], q)) for q in QS])
    fp = np.mean(fps, axis=0)
    ok = np.flatnonzero(fp <= 0.02)
    return float(QS[ok[0]] if len(ok) else QS[-1]), fp.tolist()


out = {}
NESTED = "--nested" in sys.argv
for model in ("F21", "lr", "hgb"):
    agg = {"single": [], "res": [], "sim": [], "all": [], "unequal": [], "triple": []}; per = {}
    for m in sorted(set(cell)):
        tr, te = cell != m, cell == m
        s_tr, s_te = scorer(model, tr, te)
        qlev = nested_q(model, tr)[0] if NESTED else 0.98
        thr = np.quantile(s_tr[kind[tr] == "single"], qlev)
        # matched comparison: threshold at exactly 2% false two on the held-out morphology's own singles (not deployable)
        thr_m = np.quantile(s_te[kind[te] == "single"], 0.98)
        agg.setdefault("res_matched2", []).extend(list((s_te > thr_m)[(kind[te] == "pair") & res[te]]))
        agg.setdefault("sim_matched2", []).extend(list((s_te > thr_m)[(kind[te] == "pair") & res[te] & (cond[te] == "dt0")]))
        two = s_te > thr; k_te = kind[te]
        agg["single"] += list(two[k_te == "single"]); agg["res"] += list(two[(k_te == "pair") & res[te]])
        agg["sim"] += list(two[(k_te == "pair") & res[te] & (cond[te] == "dt0")]); agg["all"] += list(two[k_te == "pair"])
        agg["unequal"] += list(two[(k_te == "pair") & (cond[te] == "unequal")]); agg["triple"] += list(two[k_te == "triple"])
        per[m] = {"false_two": float(two[k_te == "single"].mean()), "recall_res": float(two[(k_te == "pair") & res[te]].mean()) if ((k_te == "pair") & res[te]).any() else None}
    out[model] = {k: float(np.mean(v)) for k, v in agg.items()}
    out[model]["n"] = {k: len(v) for k, v in agg.items()}
    out[model]["per_morphology"] = per
    out[model]["max_morph_false_two"] = float(max(v["false_two"] for v in per.values()))
    print(model, {k: round(v, 3) for k, v in out[model].items() if isinstance(v, float)})
Path(f"reports/phase15/overlap_lomo{'_nested' if NESTED else ''}{'_' + os.environ['FAM'].replace(',', '-') if 'FAM' in os.environ else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
