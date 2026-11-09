"""Phase 12.4: expanded overlap benchmark. Usage: run_12_4.py build <pool> | fit | eval <pool>

build  generate the pool (syn + bio families) and analyze every event once (data/phase12/bench_<pool>.pkl)
fit    p12dev only: fit the logistic count model (singles vs resolvable pairs), set its threshold for a 2% dev
       false-two rate, calibrate the runtime gate (keep >= 99% of the dev resolvable pairs the LR rule recovers);
       writes reports/phase12/count_model_p12.json
eval   score P11 / P12-LR / gated P12-LR on a pool; writes reports/phase12/exp12_4_<pool>.json
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase10.overlap_stress import frozen_calibration, localizer
from edge.phase11.calibration import wilson
from edge.phase12 import count_model as CM, overlap_bench as OB

mode = sys.argv[1]
CAL = frozen_calibration()


def build(pool, scale):
    t = time.time()
    evs = OB.syn_events(pool, 40 * scale) + OB.bio_events(pool, 500 * scale, 100 * scale, 100 * scale)
    locs, recs = {}, []
    for i, ev in enumerate(evs):
        E = ev.snippet.electrode_xyz_um
        key = E.tobytes()
        if key not in locs:
            locs[key] = localizer(OB._array(E))
        recs.append(CM.analyze(locs[key], ev.snippet))
        if i % 500 == 0:
            print(pool, i, len(evs), f"{time.time() - t:.0f}s", flush=True)
    for ev in evs:
        ev.snippet = None            # keep the pickle small; analysis records hold everything needed
    pickle.dump({"events": evs, "recs": recs}, open(f"data/phase12/bench_{pool}.pkl", "wb"))
    print("built", pool, len(evs), f"{time.time() - t:.0f}s")


def load(pool):
    d = pickle.load(open(f"data/phase12/bench_{pool}.pkl", "rb"))
    keep = [i for i, r in enumerate(d["recs"]) if not r["none"]]
    return [d["events"][i] for i in keep], [d["recs"][i] for i in keep], len(d["recs"]) - len(keep)


def summarize(rows, n_true):
    if not rows:
        return {"n": 0}
    full = [r for r in rows if r["n_est"] >= n_true and r["matched"] == n_true]
    errs = [e for r in rows for e in r["errs"]]
    k = len(full)
    return {"n": len(rows), "all_sources_recovered": k / len(rows), "ci95": list(wilson(k, len(rows))),
            "declared": {str(j): float(np.mean([r["n_est"] == j for r in rows])) for j in (1, 2, 3)},
            "phantom_rate": float(np.mean([r["extra"] > 0 for r in rows])),
            "missing_rate": float(np.mean([r["missed"] > 0 for r in rows])),
            "matched_err_median_um": float(np.median(errs)) if errs else None,
            "matched_err_p90_um": float(np.percentile(errs, 90)) if errs else None}


def report(evs, recs, rule, gate=None):
    rows = []
    for ev, a in zip(evs, recs):
        invoked = True if gate is None else gate.invoke(a)
        d = rule(a) if invoked else CM.finish(a, False, True, CAL)
        r = CM.score(ev, d)
        r.update(family=ev.family, kind=ev.kind, cond=ev.cond, res=ev.resolvable, dt=ev.dt, sep=ev.sep, invoked=invoked,
                 ms=(a["ms_full"] if invoked else a["ms_k1"]) + (a["ms_feat"] if gate is not None else 0.0))
        rows.append(r)
    o = {}
    for fam in ("all", "syn", "bio"):
        R = [r for r in rows if fam == "all" or r["family"] == fam]
        S = [r for r in R if r["kind"] == "single"]; P = [r for r in R if r["kind"] == "pair"]; T = [r for r in R if r["kind"] == "triple"]
        res = [r for r in P if r["res"]]
        nf = sum(r["n_est"] >= 2 for r in S)
        o[fam] = {"singles": {"n": len(S), "false_two": nf / max(len(S), 1), "false_two_ci95": list(wilson(nf, len(S))),
                              "by_cond": {c: {"n": sum(r["cond"] == c for r in S),
                                              "false_two": float(np.mean([r["n_est"] >= 2 for r in S if r["cond"] == c]))}
                                          for c in sorted({r["cond"] for r in S})},
                              "primary_err_median": float(np.median([r["primary_err"] for r in S])) if S else None},
                  "resolvable_pairs": summarize(res, 2), "all_pairs": summarize(P, 2),
                  "simultaneous_resolvable": summarize([r for r in res if r["dt"] == 0], 2),
                  "simultaneous_all": summarize([r for r in P if r["dt"] == 0], 2),
                  "triples": summarize(T, 3), "triples_resolvable": summarize([r for r in T if r["res"]], 3),
                  "pair_err_median_resolvable_recovered": float(np.median([e for r in res if r["matched"] == 2 for e in r["errs"]])) if res else None}
    o["invoked_fraction"] = float(np.mean([r["invoked"] for r in rows]))
    o["invoked_fraction_singles"] = float(np.mean([r["invoked"] for r in rows if r["kind"] == "single"]))
    o["ms_median"] = float(np.median([r["ms"] for r in rows]))
    o["ms_median_singles"] = float(np.median([r["ms"] for r in rows if r["kind"] == "single"]))
    return o, rows


if mode == "build":
    pool = sys.argv[2]
    build(pool, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
elif mode == "fit":
    evs, recs, _ = load("p12dev")
    X = np.array([a["x"] for a in recs]); kind = np.array([e.kind for e in evs]); res = np.array([e.resolvable for e in evs])
    use = (kind == "single") | ((kind == "pair") & res)
    model = CM.LogisticCount().fit(X[use], (kind[use] == "pair").astype(int), X[kind == "single"], target_fp=0.02)
    _, rows = report(evs, recs, lambda a: CM.decide_lr(a, CAL, model))
    pos = [a for a, r, e in zip(recs, rows, evs) if e.kind == "pair" and e.resolvable and r["matched"] == 2 and r["n_est"] >= 2]
    neg = [a for a, e in zip(recs, evs) if e.kind == "single"]
    gate = CM.Gate().calibrate(pos, neg, keep=0.99)
    Path("reports/phase12/count_model_p12.json").write_text(json.dumps({"versions": versions(), "trained_on": "p12dev",
        "n_train": int(use.sum()), "logistic": model.to_json(), "gate": gate.thr,
        "rule": "two sources iff p > thr (dev single false-two 2%); free vs shared by normalized drop per extra dof; "
                "P11 merge and axial-boundary checks unchanged; gate keeps >= 99% of dev resolvable pairs recovered by the rule"},
        indent=1) + "\n")
    print(json.dumps(model.to_json()), gate.thr)
elif mode == "eval":
    pool = sys.argv[2]
    evs, recs, n_none = load(pool)
    cm = json.loads(Path("reports/phase12/count_model_p12.json").read_text())
    model = CM.LogisticCount.from_json(cm["logistic"]); gate = CM.Gate(cm["gate"])
    out = {"versions": versions(), "pool": pool, "n_events": len(evs), "no_signal_events": n_none, "rules": {}}
    rows_all = {}
    for name, rule, g in (("P11", lambda a: CM.decide_p11(a, CAL), None), ("P12-LR", lambda a: CM.decide_lr(a, CAL, model), None),
                          ("P12-LR+gate", lambda a: CM.decide_lr(a, CAL, model), gate)):
        out["rules"][name], rows_all[name] = report(evs, recs, rule, g)
        o = out["rules"][name]["all"]
        print(f"{pool} {name:12s} false2 {o['singles']['false_two']:.3f} {np.round(o['singles']['false_two_ci95'], 3)} "
              f"res {o['resolvable_pairs']['all_sources_recovered']:.3f} simul {o['simultaneous_resolvable']['all_sources_recovered']:.3f} "
              f"allpairs {o['all_pairs']['all_sources_recovered']:.3f} trip {o['triples']['all_sources_recovered']:.3f} "
              f"pair-err {o['pair_err_median_resolvable_recovered']:.1f} invoked {out['rules'][name]['invoked_fraction']:.3f} "
              f"ms {out['rules'][name]['ms_median']:.2f}", flush=True)
        for fam in ("syn", "bio"):
            f = out["rules"][name][fam]
            print(f"     {fam}: false2 {f['singles']['false_two']:.3f} res {f['resolvable_pairs']['all_sources_recovered']:.3f} "
                  f"simul {f['simultaneous_resolvable']['all_sources_recovered']:.3f} trip {f['triples']['all_sources_recovered']:.3f} "
                  f"conds {json.dumps({c: round(v['false_two'], 3) for c, v in f['singles']['by_cond'].items()})}")
    Path(f"reports/phase12/exp12_4_{pool}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    pickle.dump(rows_all, open(f"data/phase12/rows12_4_{pool}.pkl", "wb"))
