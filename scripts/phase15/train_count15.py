"""Train the research count model B-P15 (gradient-boosted trees on line20 + ms3 spatially whitened statistics and
temporal features) on all 13 BBP morphologies (Phase 14 dev/select/final NP1 pools) and export it for the numpy
evaluator (edge/phase15/trees.py). The threshold level is the nested leave-one-morphology-out quantile chosen in
scripts/phase15/overlap_lomo.py (2% target); deployments calibrate the actual threshold per recording on
development-morphology singles in that recording's background. Run with data/ibl-venv and FAM=line20,ms3.
Writes reports/phase15/count_b_p15.json."""
import json, os, sys
os.environ.setdefault("FAM", "line20,ms3")
sys.argv = [sys.argv[0]]
src = open("scripts/phase15/overlap_lomo.py").read().split("out = {}")[0]
exec(src)
from sklearn.ensemble import HistGradientBoostingClassifier
mu, sd = X.mean(0), X.std(0) + 1e-9
use = kind != "triple"
clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(((X - mu) / sd)[use], y[use])
q, fp = nested_q("hgb", np.ones(len(X), bool))
trees = []
for it in clf._predictors:
    nd = it[0].nodes
    trees.append({"feature": nd["feature_idx"].astype(int).tolist(), "thr": nd["num_threshold"].astype(float).tolist(),
                  "left": nd["left"].astype(int).tolist(), "right": nd["right"].astype(int).tolist(), "is_leaf": nd["is_leaf"].astype(bool).tolist(),
                  "value": nd["value"].astype(float).tolist(), "missing_left": nd["missing_go_to_left"].astype(bool).tolist()})
base = float(np.ravel(clf._baseline_prediction)[0])
d = {"features": list(FAM), "mu": mu.tolist(), "sd": sd.tolist(), "baseline": base, "trees": trees, "quantile": q, "nested_fp_by_quantile": fp,
     "quantiles": QS.tolist()}
json.dump(d, open("reports/phase15/count_b_p15.json", "w"))
from edge.phase15.trees import Forest
f = Forest(d); pr = clf.predict_proba(((X - mu) / sd)[:200])[:, 1]
print("quantile", q, "max |numpy - sklearn|", max(abs(f.prob(x) - p) for x, p in zip(X[:200], pr)))
