"""Phase 12.4 real-data proxy for source counting (no truth). Usage: run_12_4_real.py dev|select

Matched sorter events (11A rule) are labelled by the sorter's own spike trains, fixed before evaluation:
  isolated    no other sorter spike (any unit) within +-1.5 ms and 150 um of the event's sorter channel
  coincident  another unit's spike within +-0.5 ms and within 100 um (a likely overlap)
Unsorted background spikes are invisible to these labels, so the isolated rate is an upper-bound-like proxy of a
false-two rate, and neither label is ground truth. A useful count rule must declare two more often on coincident
than on isolated events. Up to 40 events per window per label (seeded).
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase10.overlap_stress import frozen_calibration, localizer
from edge.phase11.calibration import wilson
from edge.phase11.referencing import load_events, match_ab, snippet_at
from edge.phase12 import count_model as CM
from edge.phase12.splits import split_windows
from scripts.phase5.data import load_window, window_path

split = sys.argv[1]
CAL = frozen_calibration()
cm = json.loads(Path("reports/phase12/count_model_p12.json").read_text())
LR = CM.LogisticCount.from_json(cm["logistic"])
rng = np.random.default_rng(124)
rec = defaultdict(list)
t0 = time.time()
for w in split_windows(split):
    p = window_path(w["pid"], w["t0_s"], w["duration_s"])
    if not p.exists():
        continue
    we = load_events(w, load_window(p))
    E = we.array.xyz_um(); loc = localizer(we.array)
    matched = sorted({r for _, r, _ in match_ab(we, we.det_sample)})
    s_all, c_all, u_all = we.sorter_sample, we.sorter_channel, we.sorter_cluster
    lab = {}
    for i in matched:
        dt = np.abs(s_all - s_all[i]); d = np.linalg.norm(E[c_all] - E[c_all[i]], axis=1)
        other = np.arange(len(s_all)) != i
        if not np.any(other & (dt <= 45) & (d <= 150)):
            lab[i] = "isolated"
        elif np.any(other & (u_all != u_all[i]) & (dt <= 15) & (d <= 100)):
            lab[i] = "coincident"
    for L in ("isolated", "coincident"):
        idx = [i for i, l in lab.items() if l == L]
        for i in (rng.choice(idx, min(40, len(idx)), replace=False) if idx else []):
            sn = snippet_at(we, int(s_all[i]), int(c_all[i]), int(i))
            if sn is None:
                continue
            a = CM.analyze(loc, sn)
            if a["none"]:
                continue
            rec[L].append({"session": w["session"], "p11": CM.decide_p11(a, CAL).n, "lr": CM.decide_lr(a, CAL, LR).n,
                           "p_lr": float(LR.prob(a["x"])[0]), "chi1": a["chi1"], "snr": a["snr_obs"]})
    print(w["session"], {k: len(v) for k, v in rec.items()}, f"{time.time() - t0:.0f}s", flush=True)
out = {"versions": versions(), "split": split, "labels": "isolated: no sorter spike +-1.5 ms / 150 um; coincident: other unit +-0.5 ms / 100 um"}
for L, R in rec.items():
    o = {"n": len(R)}
    for rule in ("p11", "lr"):
        k = sum(r[rule] >= 2 for r in R)
        o[f"{rule}_two_rate"] = k / max(len(R), 1); o[f"{rule}_two_ci95"] = list(wilson(k, len(R)))
    o["chi1_median"] = float(np.median([r["chi1"] for r in R])); o["p_lr_median"] = float(np.median([r["p_lr"] for r in R]))
    o["per_session_lr"] = {s: float(np.mean([r["lr"] >= 2 for r in R if r["session"] == s])) for s in sorted({r["session"] for r in R})}
    out[L] = o
from scipy.stats import mannwhitneyu
pi = [r["p_lr"] for r in rec["isolated"]]; pc = [r["p_lr"] for r in rec["coincident"]]
out["auc_lr_coincident_vs_isolated"] = float(mannwhitneyu(pc, pi).statistic / (len(pc) * len(pi))) if pi and pc else None
Path(f"reports/phase12/exp12_4_real_{split}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=1, default=float))
