"""Phase 15 Track D cross-animal decoding experiments (data/ibl-venv).

Usage: run_xanimal.py <mode> '<spec json>' <tag>
  mode dev     leave-one-mouse-out over the 11 development mice (Phase 13-14 dev + select)
  mode select  train on the 11 development mice, test each Phase 14 final mouse (6; a used benchmark, selection only)
  mode final   train on all 17 Phase 13-14 mice, test each Phase 15 final mouse once (requires frozen_p15.json)
Spec fields: edge/phase15/xanimal.Spec. Writes reports/phase15/xa_<mode>_<tag>.json with per-mouse balanced accuracy,
chance, shuffled-training-label control and a per-mouse paired comparison vector for later tests.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from edge.phase15 import xanimal as X
from scripts.phase14.run_vis14 import labels, counts_of
from sklearn.metrics import balanced_accuracy_score

TASKS = ("SG_sf5", "SG_orientation6", "DG_orientation8", "GB_x", "GB_y", "NS_image118")


def mice():
    s14 = json.loads(Path("reports/phase14/allen_split14.json").read_text())
    inv14 = json.loads(Path("reports/phase14/allen_inventory14.json").read_text())
    ok = lambda s: inv14.get(s, {}).get("status") == "ok"
    tr = [s for s in s14["train_candidates"] if ok(s)][:6]; fi = [s for s in s14["final_candidates"] if ok(s)][:6]
    dev = [s for s in s14["dev"] + s14["select_p13"] if ok(s)] + tr
    out = {"dev": dev, "select": fi, "final": []}
    p = Path("reports/phase15/allen_split15.json")
    if p.exists():
        s15 = json.loads(p.read_text()); inv = Path("reports/phase15/allen_inventory15.json")
        inv15 = json.loads(inv.read_text()) if inv.exists() else {}
        out["final"] = [s for s in s15["final_candidates"] if inv15.get(s, {}).get("status") == "ok"][:6]
        out["train15"] = [s for s in s15["train_candidates"] if inv15.get(s, {}).get("status") == "ok"]
    return out


_cache = {}


def load(s):
    if s not in _cache:
        from scripts.phase14.run_vis14 import load as L
        _cache[s] = L(s)
    return _cache[s]


def block(s, task):
    d = load(s); lab = labels(d)
    if task not in lab:
        return None
    key, m, y = lab[task]
    return counts_of(d, key)[m], d, y[m]


_fcache = {}


def feat(s, task, spec, centres):
    """Per-mouse feature matrix (cached when it does not depend on fold-specific centres)."""
    key = (s, task) if centres is None else None
    if key is not None and key in _fcache:
        return _fcache[key]
    b = block(s, task)
    if b is None:
        return None
    out = (X.features(b[0], b[1], spec, centres), b[2])
    if key is not None:
        _fcache[key] = out
        _cache.pop(s, None)            # raw counts are large; features are what is reused
    return out


def run(mode, spec: X.Spec, tag):
    M = mice()
    if mode == "dev":
        folds = [([s for s in M["dev"] if s != t], t) for t in M["dev"]]
    elif mode == "select":
        folds = [(M["dev"], t) for t in M["select"]]
    else:
        if not Path("reports/phase15/frozen_p15.json").exists():
            raise SystemExit("frozen_p15.json required")
        folds = [(M["dev"] + M["select"] + M.get("train15", []), t) for t in M["final"]]
    if spec.extra.get("n_train"):
        rs = np.random.default_rng(spec.extra.get("seed", 0))
        folds = [(sorted(rs.choice(tr, min(spec.extra["n_train"], len(tr)), replace=False).tolist()), te) for tr, te in folds]
    out = {"mode": mode, "spec": spec.__dict__, "tag": tag, "tasks": {}}
    for task in TASKS:
        res = {}
        for tr, te in folds:
            bte = block(te, task)
            if bte is None:
                continue
            centres = None
            if "fclust" in spec.groups:
                P = np.vstack([X.fclust_profiles(block(s, task)[0], spec) for s in tr if block(s, task) is not None])
                centres = X.kmeans(P, spec.k_fclust, seed=0)
            Ftr, ytr = [], []
            for s in tr:
                f = feat(s, task, spec, centres)
                if f is None:
                    continue
                Ftr.append(f[0]); ytr.append(f[1])
            Xtr, ytr = np.vstack(Ftr), np.concatenate(ytr)
            Xte, yte = feat(te, task, spec, centres)
            if spec.coral:
                Xte = X.coral(Xtr, Xte)
            p, P, cls = X.fit_predict(Xtr, ytr, Xte, spec)
            r = {"bacc": float(balanced_accuracy_score(yte, p)), "chance": 1.0 / len(np.unique(yte)), "n_train_mice": len(Ftr),
                 "n_features": int(Xtr.shape[1]), "correct": (p == yte).astype(int).tolist() if task != "NS_image118" else None}
            if spec.extra.get("controls"):
                ysh = np.random.default_rng(4).permutation(ytr)
                r["shuffled_train_labels"] = float(balanced_accuracy_score(yte, X.fit_predict(Xtr, ysh, Xte, spec)[0]))
            res[te] = r
        out["tasks"][task] = res
        _cache.clear() if len(_cache) > 40 else None
        b = [v["bacc"] for v in res.values()]
        print(tag, task, "mean", round(float(np.mean(b)), 3), "min", round(float(np.min(b)), 3), "chance", round(list(res.values())[0]["chance"], 3), flush=True)
    out["summary"] = {t: float(np.mean([v["bacc"] for v in r.values()])) for t, r in out["tasks"].items() if r}
    Path(f"reports/phase15/xa_{mode}_{tag}.json").write_text(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    mode, spec, tag = sys.argv[1], X.Spec(**json.loads(sys.argv[2])), sys.argv[3]
    t = time.time(); run(mode, spec, tag); print("seconds", round(time.time() - t))
