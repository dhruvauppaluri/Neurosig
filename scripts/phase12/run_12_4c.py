"""Phase 12.4c: dictionary null model for source counting. Usage: run_12_4c.py bio <pool> | real

bio   pool p12dev: leave-template-out (atoms of the event's own templates excluded), calibrate tau_F at a 2% single
      false-two rate on bio singles; pools p12select / p12final: the dictionary is built from the dev templates only
      (the pool's templates are never atoms). Reports false-two (Wilson CI), recall, simultaneous, triples.
real  dictionary from real dev units (sessions A-C, matched events, >= 30 events, robust mean waveform, source = the
      l20_robust pooled position); tau_F calibrated on dev *isolated* events (no other sorter spike within +-1.5 ms and
      150 um, a proxy for single-source events). On the select sessions D-P: declared-two rate on isolated vs
      coincident events (another sorter unit within +-0.5 ms and 100 um) for P11 and the dictionary rule.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase11.calibration import wilson
from edge.phase11 import forward_models as FM
from edge.phase12 import biosim as B, dictionary as D, overlap_bench as OB
from edge.phase10.overlap_stress import match
from edge.phase8.multisource import _gain  # noqa: F401
from edge.phase8.refine import channel_noise

mode = sys.argv[1]
OUT = Path("reports/phase12"); DAT = Path("data/phase12")


def bio_dictionary():
    atoms = []
    for k in B.template_pool("dev"):
        W, E, tr, t0, pk = B.template_view(k)
        a = D.make_atom(W, E, t0, tr, str(k))
        if a is not None:
            atoms.append(a)
    return D.Dictionary(atoms)


def run_bio(pool):
    dic = bio_dictionary()
    evs = [e for e in OB.syn_events(pool, 40) + OB.bio_events(pool, 500, 100, 100)]
    rows, cache, t0 = [], {}, time.time()
    for i, ev in enumerate(evs):
        sn = ev.snippet; v = np.asarray(sn.voltages_uv, float); E = sn.electrode_xyz_um
        sig = channel_noise(v)
        ex = frozenset(str(k) for k in ev.tids) if (pool == "p12dev" and ev.family == "bio") else frozenset()
        t = time.perf_counter()
        f = dic.fit(v, sn.n_pre, sig, E, exclude=ex)
        ms = (time.perf_counter() - t) * 1e3
        p2 = D.demixed_positions(f, v, sig, E, cache) if f.get("two") else None
        a1 = -(v[sn.n_pre] - np.median(v[:max(sn.n_pre - 12, 4)], axis=0))
        f1 = FM.fit(FM.Model("line20"), a1, sig, E, cache=cache)
        rows.append({"family": ev.family, "kind": ev.kind, "cond": ev.cond, "res": ev.resolvable, "dt": ev.dt, "F": f["F"],
                     "pos1": f1.theta[:2].copy(), "pos1_dict": f["pos1_dict"], "pos2": p2, "truth": ev.sources, "ms": ms})
        if i % 500 == 0:
            print(pool, i, len(evs), f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(rows, open(DAT / f"rows12_4c_{pool}.pkl", "wb"))
    return rows


def score(rows, tau):
    out = {}
    for fam in ("all", "syn", "bio"):
        R = [r for r in rows if fam == "all" or r["family"] == fam]
        def dec(r):
            return 2 if (r["F"] > tau and r["pos2"] is not None) else 1
        S = [r for r in R if r["kind"] == "single"]
        nf = sum(dec(r) == 2 for r in S)
        def rec(L, K):
            ok = 0; errs = []
            for r in L:
                est = r["pos2"] if dec(r) == 2 else r["pos1"][None]
                m = match(r["truth"], est)
                ok += (m["matched"] == K and len(est) >= K); errs += m["errs"] if m["matched"] == K else []
            return {"n": len(L), "all_sources_recovered": ok / max(len(L), 1), "ci95": list(wilson(ok, len(L))) if L else None,
                    "err_median": float(np.median(errs)) if errs else None}
        P = [r for r in R if r["kind"] == "pair"]; res = [r for r in P if r["res"]]
        out[fam] = {"false_two": nf / max(len(S), 1), "false_two_ci95": list(wilson(nf, len(S))) if S else None, "n_singles": len(S),
                    "by_cond": {c: float(np.mean([dec(r) == 2 for r in S if r["cond"] == c])) for c in sorted({r["cond"] for r in S})},
                    "resolvable_pairs": rec(res, 2), "all_pairs": rec(P, 2),
                    "simultaneous_resolvable": rec([r for r in res if r["dt"] == 0], 2),
                    "triples": rec([r for r in R if r["kind"] == "triple"], 3) if fam != "syn" else None}
    out["ms_median"] = float(np.median([r["ms"] for r in rows]))
    return out


if mode == "bio":
    pool = sys.argv[2]
    p = DAT / f"rows12_4c_{pool}.pkl"
    rows = pickle.load(open(p, "rb")) if p.exists() else run_bio(pool)
    cfg = OUT / "dictionary_count_p12.json"
    if pool == "p12dev":
        tau = float(np.quantile([r["F"] for r in rows if r["kind"] == "single" and r["family"] == "bio"], 0.98))
        cfg.write_text(json.dumps({"tau_F_bio": tau, "calibrated_on": "p12dev bio singles (leave-template-out), 2% false two",
                                   "dictionary": "36 dev MEArec templates + lateral mirrors"}, indent=1) + "\n")
    tau = json.loads(cfg.read_text())["tau_F_bio"]
    o = {"versions": versions(), "pool": pool, "tau_F": tau, **score(rows, tau)}
    (OUT / f"exp12_4c_bio_{pool}.json").write_text(json.dumps(o, indent=1, default=float) + "\n")
    for fam in ("all", "syn", "bio"):
        f = o[fam]
        print(pool, fam, f"false2 {f['false_two']:.3f} {np.round(f['false_two_ci95'], 3)} res {f['resolvable_pairs']['all_sources_recovered']:.3f} "
              f"simul {f['simultaneous_resolvable']['all_sources_recovered']:.3f} allpairs {f['all_pairs']['all_sources_recovered']:.3f} "
              f"err {f['resolvable_pairs']['err_median']} conds {f['by_cond']}" + (f" trip {f['triples']['all_sources_recovered']:.3f}" if f.get('triples') else ""))
    print("ms", o["ms_median"])
