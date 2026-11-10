"""Phase 15 final cross-animal comparison: D-P15 vs the Phase 14 reference spec on the same 6 final mice (paired by
mouse), plus the pre-registered secondary per-task routing (D-P15 / e11). Per task: mean balanced accuracy, per-mouse
paired difference, bootstrap 95% CI over mice (seed 0, 10,000 resamples), number of mice improved, exact sign-test p.
Writes reports/phase15/xa_final_comparison.json."""
import json
from math import comb
from pathlib import Path

import numpy as np

R = Path("reports/phase15")
fr = json.loads((R / "frozen_p15.json").read_text())
L = lambda t: json.loads((R / f"xa_final_{t}.json").read_text())  # noqa: E731
dp, ref = L("final_dp15"), L("final_ref14")
e11 = L("final_e11") if (R / "xa_final_final_e11.json").exists() else None
route = fr["vision"]["D-P15-route_secondary"]
rng = np.random.default_rng(0)


def paired(a, b):
    m = sorted(set(a) & set(b)); d = np.array([a[k]["bacc"] - b[k]["bacc"] for k in m])
    bs = [float(np.mean(rng.choice(d, len(d)))) for _ in range(10000)]
    npos, n = int((d > 0).sum()), int((d != 0).sum())
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(npos, n - npos) + 1)) / 2 ** n) if n else 1.0
    return {"n_mice": len(m), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_improved": npos, "sign_test_p": p, "per_mouse": {k: float(x) for k, x in zip(m, d)}}


out = {"frozen_hash": fr["hash"], "tasks": {}}
for t in dp["tasks"]:
    a, b = dp["tasks"][t], ref["tasks"][t]
    r = {"D-P15": float(np.mean([v["bacc"] for v in a.values()])), "reference_p14_spec": float(np.mean([v["bacc"] for v in b.values()])),
         "chance": next(iter(a.values()))["chance"], "D-P15_minus_reference": paired(a, b)}
    if e11 is not None:
        src = a if route[t] == "e7n_fclust" else e11["tasks"][t]
        r["route_secondary"] = float(np.mean([v["bacc"] for v in src.values()]))
        r["route_minus_reference"] = paired(src, b)
    out["tasks"][t] = r
(R / "xa_final_comparison.json").write_text(json.dumps(out, indent=1) + "\n")
for t, r in out["tasks"].items():
    d = r["D-P15_minus_reference"]
    print(f"{t:16s} D-P15 {r['D-P15']:.3f} ref {r['reference_p14_spec']:.3f} diff {d['mean_diff']:+.3f} ci {d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f} "
          f"improved {d['n_improved']}/{d['n_mice']} p {d['sign_test_p']:.3f}" + (f" | route {r['route_secondary']:.3f}" if "route_secondary" in r else ""))
