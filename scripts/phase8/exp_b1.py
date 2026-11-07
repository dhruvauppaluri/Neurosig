"""K=2 usefulness on dev: per-source Hungarian errors and primary-source error."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from edge.benchmark_v3 import _array
from edge.phase8.multisource import SpaceTimeFitter
from edge.phase8.synth import make_events

array = _array("linear")
events = make_events("dev", n_per_cell=40)
f = SpaceTimeFitter(array)
rows = []
for e in events:
    m = f.fit(e.snippet)
    truth = np.array([[np.hypot(p[0], p[2]), p[1]] for p in e.sources_xyz_um])
    est2 = np.array([[s.x, s.y] for s in m.sources2])
    est1 = np.array([[s.x, s.y] for s in m.sources1])
    D = np.linalg.norm(truth[:, None] - est2[None], axis=2)
    r, c = linear_sum_assignment(D)
    sep = np.linalg.norm(truth[0] - truth[1]) if len(truth) > 1 else np.nan
    rows.append((e.noise_uv, e.contaminated, m.delta, np.linalg.norm(truth[0] - est1[0]),
                 D[r, c].mean(), sep))
R = np.array(rows, float)
for nz in (2.0, 10.0, 25.0):
    k = (R[:, 0] == nz) & (R[:, 1] == 1)
    print(nz, "contam n", k.sum(), "K1 err to src1 median", np.median(R[k, 3]).round(1),
          "K2 hungarian mean-err median", np.median(R[k, 4]).round(1),
          "frac delta>150", (R[k, 2] > 150).mean().round(2))
    k0 = (R[:, 0] == nz) & (R[:, 1] == 0)
    print("   single: K1 median", np.median(R[k0, 3]).round(1), "frac delta>150", (R[k0, 2] > 150).mean().round(2))
