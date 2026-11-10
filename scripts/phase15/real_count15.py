"""Phase 15 real-recording count proxy (sorter used only to define reference groups): P14-C (frozen Phase 14) and
B-P15 (research) on isolated vs sorter-coincident events and pooled good-unit templates, per session, with
per-session label-free threshold self-calibration. Usage: real_count15.py <dev|select|final>.
Writes reports/phase15/real_count15_<split>.json."""
import json, pickle, sys, copy
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase12 import pooling as PL
from edge.phase12.splits import split_windows
from edge.phase14 import bench14 as B
from edge.phase14.pipeline14 import P14Config, SessionContext, run_event
from edge.phase14.realnoise import session_noise
from edge.phase14.neuron_model import NoiseModel
from edge.phase15.count15 import CountB

split = sys.argv[1]
fr = json.loads(Path("reports/phase14/frozen_p14.json").read_text())
cfg = P14Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in fr["p14"].items()})
if split == "final":
    wins = json.loads(Path("reports/phase15/split_manifest_phase15.json").read_text())["windows"]; cdir = "data/phase15"
else:
    wins = split_windows(split); cdir = "data/phase12"
by = defaultdict(list)
for w in wins:
    by[w["session"]].append(w)
qual = {}
for p in ("data/phase12/cluster_info.pkl", "data/phase14/cluster_info14.pkl", "data/phase15/cluster_info15.pkl"):
    if Path(p).exists():
        for k, v in pickle.load(open(p, "rb")).items():
            qual[k] = dict(zip(v["cluster_id"], v["label"]))
rng = np.random.default_rng(20271561)
out = {"split": split, "sessions": {}}
sessions = sorted(by) if split != "select" else ["D", "E", "H", "L", "M", "O", "P"]
for s in sessions:
    path = Path(cdir) / f"sess_{s}.npz"
    if not path.exists():
        out["sessions"][s] = {"status": "missing_dataset"}; continue
    se = PL.load_session(path, None)
    native, axial, mad = session_noise(by[s])
    cal = [e.snippet for e in B.generate("p14dev", 60, 0, 0, calibration=True, noise_override=(axial, mad))]
    ctx = SessionContext(native, cfg); ctx.thr2 = SessionContext(axial, cfg).calibrate(cal)
    cb = CountB(NoiseModel(native)); cb_cal = CountB(NoiseModel(axial)); cb.thr = cb_cal.calibrate(cal)
    E = se.E[:, :2]; t = se.t_s; ch = se.sorter_ch
    iso, coin = [], []
    for i in range(len(t)):
        near = np.flatnonzero((np.abs(t - t[i]) <= 1.5e-3) & (np.arange(len(t)) != i))
        dch = np.linalg.norm(E[ch[near]] - E[ch[i]], axis=1) if len(near) else np.array([])
        if not np.any(dch <= 100):
            iso.append(i)
        if len(near) and np.any((np.abs(t[near] - t[i]) <= 0.5e-3) & (se.unit[near] != se.unit[i]) & (dch >= 40) & (dch <= 150)):
            coin.append(i)
    iso = list(rng.choice(iso, min(300, len(iso)), replace=False)) if iso else []
    coin = list(rng.choice(coin, min(300, len(coin)), replace=False)) if coin else []
    r = {}
    for nm, idx in (("isolated", iso), ("coincident", coin)):
        a = [run_event(ctx, se.snippet(int(i)), with_region=False)["n"] >= 2 for i in idx]
        b = [cb.score(se.snippet(int(i))) > cb.thr for i in idx]
        r[nm] = {"n": len(idx), "two_p14": float(np.mean(a)) if a else None, "two_bp15": float(np.mean(b)) if b else None}
    out["sessions"][s] = {"thr_p14": ctx.thr2, "thr_bp15": cb.thr, **r}
    print(s, out["sessions"][s], flush=True)
ok = [v for v in out["sessions"].values() if "isolated" in v and v["isolated"]["two_p14"] is not None]
out["summary"] = {f"{g}_{m}": float(np.median([v[g][m] for v in ok if v[g][m] is not None])) for g in ("isolated", "coincident") for m in ("two_p14", "two_bp15")}
print(out["summary"])
Path(f"reports/phase15/real_count15_{split}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
