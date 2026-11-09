"""Phase 12.6: runtime. Usage: run_12_6.py gate | stream <split> [max_windows]

gate    fit the cheap pre-screen on p12dev (target: events the P12-LR rule declares two, keep >= 99% of them), then
        evaluate on p12select: invoked fraction, lost two-source decisions, count metrics with and without the gate.
stream  the frozen P12 event pipeline on real detector-v2.0 events of a split, in time order: E3 alignment, mean3
        amplitudes, FastLine20 + identifiability area, gate, and (if invoked) the Phase 9/10 K<=2 analysis + P12-LR.
        Per-event latency, invoked fraction, sustained single-core throughput, a FIFO queue simulation at the real
        arrival times (buffer 64), and peak traced memory (separate pass). Single process, one core, this machine.
"""
import json, pickle, sys, time, tracemalloc
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase8.refine import channel_noise
from edge.phase10.overlap_stress import frozen_calibration, localizer
from edge.phase11.calibration import wilson
from edge.phase12 import count_model as CM, fast as FA, overlap_bench as OB
from edge.phase10.overlap_stress import match

CAL = frozen_calibration()
cm = json.loads(Path("reports/phase12/count_model_p12.json").read_text())
LR = CM.LogisticCount.from_json(cm["logistic"])


def bench_rows(pool):
    d = pickle.load(open(f"data/phase12/bench_{pool}.pkl", "rb"))
    evs = OB.syn_events(pool, 40) + OB.bio_events(pool, 500, 100, 100)       # deterministic regeneration (snippets)
    assert len(evs) == len(d["events"])
    fls, rows = {}, []
    for ev, a in zip(evs, d["recs"]):
        if a["none"]:
            continue
        sn = ev.snippet; v = np.asarray(sn.voltages_uv, float); E = sn.electrode_xyz_um
        fl = fls.setdefault(E.tobytes(), FA.FastLine20(E))
        th, ar, f = FA.cheap_features(v, sn.n_pre, fl, channel_noise(v))
        dec = CM.decide_lr(a, CAL, LR)
        rows.append({"ev": ev, "a": a, "f": f, "lr_two": dec.n >= 2, "dec": dec})
    return rows


def metrics(rows, gate=None):
    out = {"singles": [], "res": [], "simul": [], "inv": []}
    for r in rows:
        inv = True if gate is None else gate.invoke(r["f"])
        d = r["dec"] if inv else CM.finish(r["a"], False, True, CAL)
        ev = r["ev"]; m = match(ev.sources, d.pos)
        out["inv"].append(inv)
        if ev.kind == "single":
            out["singles"].append(d.n >= 2)
        elif ev.kind == "pair" and ev.resolvable:
            ok = m["matched"] == 2 and d.n >= 2
            out["res"].append(ok)
            if ev.dt == 0:
                out["simul"].append(ok)
    k = int(np.sum(out["singles"]))
    return {"false_two": k / len(out["singles"]), "false_two_ci95": list(wilson(k, len(out["singles"]))),
            "res_recall": float(np.mean(out["res"])), "simultaneous": float(np.mean(out["simul"])),
            "invoked_fraction": float(np.mean(out["inv"])),
            "invoked_singles": float(np.mean([i for i, r in zip(out["inv"], rows) if r["ev"].kind == "single"]))}


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        dev = bench_rows("p12dev")
        X = np.array([r["f"] for r in dev]); y = np.array([r["lr_two"] for r in dev], int)
        m = CM.LogisticCount().fit(X, y, X[y == 0], target_fp=0.5)
        p = m.prob(X)
        thr = float(np.quantile(p[y == 1], 0.01))                      # keep >= 99% of dev LR-two events
        g = {"features": list(FA.GATE_FEATURES), "mu": m.mu.tolist(), "sd": m.sd.tolist(), "w": m.w.tolist(), "thr": thr,
             "target": "P12-LR declares two (p12dev)", "keep": 0.99}
        Path("reports/phase12/gate_p12.json").write_text(json.dumps(g, indent=1) + "\n")
        gate = FA.CheapGate(g)
        sel = bench_rows("p12select")
        out = {"versions": versions(), "gate": g, "dev": {"ungated": metrics(dev), "gated": metrics(dev, gate)},
               "select": {"ungated": metrics(sel), "gated": metrics(sel, gate)}}
        lost = [r for r in sel if r["lr_two"] and not gate.invoke(r["f"])]
        out["select"]["lost_lr_two_fraction"] = len(lost) / max(sum(r["lr_two"] for r in sel), 1)
        Path("reports/phase12/exp12_6_gate.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k in ("dev", "select")}, indent=1, default=float))

    elif mode == "stream":
        import multiprocessing as mp
        from edge.phase12.splits import split_windows
        from scripts.phase5.data import window_path
        split = sys.argv[2]; maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
        wins = [w for w in split_windows(split) if window_path(w["pid"], w["t0_s"], w["duration_s"]).exists()][:maxw]
        from scripts.phase12.stream_worker import process_window
        out = {"versions": versions(), "split": split, "windows": len(wins), "pipeline": "P12: E3 + mean3 + FastLine20 + area; "
               "fast K<=2 analysis + P12-LR on every event (gate rejected)", "workers": {}}
        for nw in (1, 2, 4):
            t = time.perf_counter()
            if nw == 1:
                res = [process_window(w) for w in wins]
            else:
                with mp.get_context("spawn").Pool(nw) as pool:
                    res = pool.map(process_window, wins)
            wall = time.perf_counter() - t
            lat = np.concatenate([r["lat"] for r in res]); single = np.concatenate([r["single_ms"] for r in res])
            rec_s = sum(r["dur_s"] for r in res); n = len(lat)
            busy = sum(r["busy_s"] for r in res)
            o = {"events": int(n), "recording_s": rec_s, "event_rate_per_s": n / rec_s,
                 "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99))},
                 "single_source_ms": {"median": float(np.median(single)), "p95": float(np.percentile(single, 95))},
                 "processing_events_per_s": n / busy * (1 if nw == 1 else 1),
                 "wall_s_including_load_and_detector": wall, "two_declared_fraction": float(np.mean(np.concatenate([r["two"] for r in res])))}
            if nw == 1:
                o["sustained_events_per_s_one_core"] = n / busy
                o["throughput_margin_one_core"] = (n / busy) / (n / rec_s)
            else:
                o["parallel_events_per_s_estimate"] = n / busy * nw
            out["workers"][str(nw)] = o
            print(nw, json.dumps(o, default=float), flush=True)
        # FIFO queue at real arrival times for 1 core (buffer 64)
        res = [process_window(w) for w in wins[:20]]
        t_off, busy_until, drops, waits, qq = 0.0, 0.0, 0, [], []
        for r in res:
            for ta, svc in zip(r["arr_s"] + t_off, r["lat"] / 1e3):
                qq = [x for x in qq if x > ta]
                if len(qq) >= 64:
                    drops += 1; continue
                st = max(ta, busy_until); busy_until = st + svc; qq.append(busy_until); waits.append(st - ta)
            t_off += r["dur_s"]
        out["queue_one_core_buffer64"] = {"events": len(waits) + drops, "drops": drops, "wait_ms_median": float(np.median(waits) * 1e3),
                                          "wait_ms_p99": float(np.percentile(waits, 99) * 1e3)}
        tracemalloc.start(); process_window(wins[0]); out["peak_traced_mib_one_window"] = tracemalloc.get_traced_memory()[1] / 2 ** 20; tracemalloc.stop()
        Path(f"reports/phase12/exp12_6_stream_{split}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=1, default=float))
