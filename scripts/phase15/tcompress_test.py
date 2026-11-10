"""Track F test: temporal SVD compression of the whitened snippet (rank r) before the ms3/line20 fits; separation
(AUC singles vs resolvable pairs), recall at 2% and runtime vs the full T = 60 statistic on p14select events."""
import sys, time, json
import numpy as np
sys.path.insert(0, ".")
from edge.phase14 import bench14 as B
from edge.phase14.neuron_model import NoiseModel, prepare
from edge.phase14.fast14 import FastSourceModel
ev = B.generate("p14select", 100, 40, 0)
nm = NoiseModel(B.noise_bank("select"))
models = {}
out = {}
for r in (None, 16, 10, 6):
    F, L, kind, res, ms = [], [], [], [], []
    for e in ev:
        E = np.asarray(e.snippet.electrode_xyz_um, float); k = E.tobytes()
        if k not in models:
            models[k] = (FastSourceModel(E, "ms3"), FastSourceModel(E, "line20"))
        pr = prepare(e.snippet, noise=nm, temporal=False)
        t = time.perf_counter()
        if r is not None:
            U, S, Vt = np.linalg.svd(pr.D, full_matrices=False)
            pr.D = U[:, :r] * S[:r]; pr.n_obs = pr.D.size
        f = models[k][0].fit(pr, 2); g = models[k][1].fit(pr, 1)
        ms.append((time.perf_counter() - t) * 1e3)
        F.append(f["F21"]); kind.append(e.kind); res.append(e.resolvable)
    F, kind, res = np.array(F), np.array(kind), np.array(res)
    s0, s1 = F[kind == "single"], F[(kind == "pair") & res]
    thr = np.quantile(s0, 0.98)
    out[str(r)] = {"auc": float(np.mean(s1[:, None] > s0[None])), "recall_at2": float(np.mean(s1 > thr)), "ms_median": float(np.median(ms))}
    print(r, out[str(r)], flush=True)
json.dump(out, open("reports/phase15/tcompress_test.json", "w"), indent=1)
