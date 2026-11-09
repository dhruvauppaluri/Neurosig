"""Phase 14.3 real-recording check of the P14 count rule (sorter used only to define reference groups).

For each session: the session NoiseModel (event-free background, native channel order) and the label-free threshold
self-calibration (development-morphology MEArec singles in this session's background). Reference groups (imperfect:
the sorter is not ground truth, and the per-unit cap of the Phase 12 cache makes coincidence detection incomplete):
  isolated     no other sorter event within +-1.5 ms on a channel within 100 um
  coincident   another unit's event within +-0.5 ms on a channel 40-150 um away
  templates    mean aligned waveform of N = 10 / 50 random events of a sorter unit with quality label 1 (whitening
               scaled by 1/N): real single-neuron waveforms at high effective SNR, the closest available real
               single-source reference
Usage: .venv/bin/python scripts/phase14/run_14_3_real.py <split> ; writes reports/phase14/exp14_3_real_<split>.json
"""
import json, pickle, sys, copy
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase12 import pooling as PL
from edge.phase12.splits import split_windows
from edge.phase14 import bench14 as B
from edge.phase14.pipeline14 import P14Config, SessionContext, run_event
from edge.phase14.realnoise import session_noise

split = sys.argv[1]
fr = json.loads(Path("reports/phase14/frozen_p14.json").read_text()) if Path("reports/phase14/frozen_p14.json").exists() else None
cfg = P14Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in fr["p14"].items()}) if fr else P14Config(k1_position="ms3")
if split == "final":
    wins_all = json.loads(Path("reports/phase14/split_manifest_phase14.json").read_text())["windows"]; cache_dir = "data/phase14"
else:
    wins_all = split_windows(split); cache_dir = "data/phase12"
by = defaultdict(list)
for w in wins_all:
    by[w["session"]].append(w)
qual = {}
for p in ("data/phase12/cluster_info.pkl", "data/phase14/cluster_info14.pkl"):
    if Path(p).exists():
        for k, v in pickle.load(open(p, "rb")).items():
            qual[k] = dict(zip(v["cluster_id"], v["label"]))
out = {"versions": versions(), "split": split, "sessions": {}}
rng = np.random.default_rng(20271461)
for s, ws in sorted(by.items()):
    path = Path(cache_dir) / f"sess_{s}.npz"
    if not path.exists():
        out["sessions"][s] = {"status": "missing_dataset"}; continue
    se = PL.load_session(path, None)
    native, axial, mad_ax = session_noise(ws)
    ctx = SessionContext(native, cfg)
    cal = B.generate("p14dev", 100, 0, 0, calibration=True, noise_override=(axial, mad_ax))
    cal_ctx = SessionContext(axial, cfg)
    ctx.thr2 = cal_ctx.calibrate([e.snippet for e in cal])
    E = se.E[:, :2]; t = se.t_s; ch = se.sorter_ch
    iso, coin = [], []
    for i in range(len(t)):
        near = np.flatnonzero((np.abs(t - t[i]) <= 1.5e-3) & (np.arange(len(t)) != i))
        dch = np.linalg.norm(E[ch[near]] - E[ch[i]], axis=1) if len(near) else np.array([])
        if not np.any(dch <= 100):
            iso.append(i)
        c = near[(np.abs(t[near] - t[i]) <= 0.5e-3) & (se.unit[near] != se.unit[i]) & (dch >= 40) & (dch <= 150)] if len(near) else []
        if len(c):
            coin.append(i)
    iso = list(rng.choice(iso, min(400, len(iso)), replace=False)) if iso else []
    decide = lambda sn: run_event(ctx, sn, with_region=False)
    r_iso = [decide(se.snippet(int(i)))["n"] >= 2 for i in iso]
    r_coin = [decide(se.snippet(int(i)))["n"] >= 2 for i in coin]
    good = [u for u in np.unique(se.unit) if qual.get(s, {}).get(int(u), 0) >= 1 and np.sum(se.unit == u) >= 50]
    tmpl = {}
    for N in (1, 10, 50):
        flags = []
        for u in good:
            idx = np.flatnonzero(se.unit == u)
            pick = rng.choice(idx, N, replace=False)
            v = se.aligned_wave(pick).mean(axis=0)
            c2 = copy.copy(ctx); c2.noise = copy.copy(ctx.noise); c2.noise.Ss = ctx.noise.Ss / N; c2.noise._ws = {}
            c2._models = ctx._models
            flags.append(run_event(c2, se.snippet(int(pick[0]), v=v), with_region=False)["n"] >= 2)
        tmpl[f"N{N}"] = {"n_units": len(flags), "false_two": float(np.mean(flags)) if flags else None}
    o = {"thr2_selfcal": ctx.thr2, "n_isolated": len(r_iso), "two_isolated": float(np.mean(r_iso)) if r_iso else None,
         "n_coincident": len(r_coin), "two_coincident": float(np.mean(r_coin)) if r_coin else None, "templates": tmpl}
    out["sessions"][s] = o
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
ok = [v for v in out["sessions"].values() if "two_isolated" in v and v["two_isolated"] is not None]
out["summary"] = {"two_isolated_median": float(np.median([v["two_isolated"] for v in ok])),
                  "two_coincident_median": float(np.median([v["two_coincident"] for v in ok if v["two_coincident"] is not None])),
                  **{f"template_false_two_{N}_median": float(np.median([v["templates"][N]["false_two"] for v in ok if v["templates"][N]["false_two"] is not None]))
                     for N in ("N1", "N10", "N50")}}
print(out["summary"])
Path(f"reports/phase14/exp14_3_real_{split}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
