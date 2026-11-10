"""Track F: batched deployment path vs the exact path on p14select (and p14dev): separation, recall at self-calibrated
2%, position errors and per-event time at batch sizes 1 / 16 / 64. Writes reports/phase15/batch_test.json."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from edge.phase14 import bench14 as B
from edge.phase14.neuron_model import NoiseModel
from edge.phase15.batch import BatchModel
out = {}
for pool in ("p14dev", "p14select"):
    ev = B.generate(pool, 150, 60, 0); cal = B.generate(pool, 150, 0, 0, calibration=True)
    nm = NoiseModel(B.noise_bank(B.POOLS[pool]["noise"]))
    models = {}
    def run(events, bs):
        res = [None] * len(events); t = time.perf_counter()
        by = {}
        for i, e in enumerate(events):
            by.setdefault(np.asarray(e.snippet.electrode_xyz_um).tobytes(), []).append(i)
        for k, idx in by.items():
            if k not in models:
                models[k] = BatchModel(np.asarray(events[idx[0]].snippet.electrode_xyz_um))
            for a in range(0, len(idx), bs):
                ii = idx[a:a + bs]
                for i, r in zip(ii, models[k].run([events[i].snippet for i in ii], nm)):
                    res[i] = r
        return res, (time.perf_counter() - t) * 1e3 / len(events)
    run(ev[:50], 16)
    o = {}
    for bs in (1, 16, 64):
        res, ms = run(ev, bs); o[f"ms_per_event_bs{bs}"] = ms
    cres, _ = run(cal, 64)
    thr = np.quantile([r["F21"] for r in cres], 0.98)
    F = np.array([r["F21"] for r in res]); kind = np.array([e.kind for e in ev]); resv = np.array([e.resolvable for e in ev]); cond = np.array([e.cond for e in ev])
    s0, s1 = F[kind == "single"], F[(kind == "pair") & resv]
    o.update(auc=float(np.mean(s1[:, None] > s0[None])), false_two=float(np.mean(s0 > thr)), recall_res=float(np.mean(s1 > thr)),
             recall_sim=float(np.mean(F[(kind == "pair") & resv & (cond == "dt0")] > thr)))
    for k in ("p1", "p1_ms3", "p1_line20"):
        d = np.array([np.linalg.norm(r[k] - e.sources[0]) for r, e in zip(res, ev) if e.kind == "single"])
        o[f"single_err_{k}"] = [float(np.median(d)), float(np.percentile(d, 90))]
    out[pool] = o; print(pool, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in o.items()}, flush=True)
json.dump(out, open("reports/phase15/batch_test.json", "w"), indent=1)
