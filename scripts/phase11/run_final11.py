"""Single-shot Phase 11 final evaluation of the frozen P11 configuration (hash in edge/phase11/configs.py) and its
comparators on the fresh final windows M-P and the fresh p11final synthetic pool. Refuses to rerun."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9.configs import p9_factory
from edge.phase9.evalset import build_events_shift
from edge.phase10.configs import p10_factory
from edge.phase10.overlap_stress import localizer, make_stress, summarize
from edge.phase11 import calibration as CAL, forward_models as F
from edge.phase11.configs import P11_HASH, P11_PARAMS, p11_factory
from edge.phase11.dropout import apply, collect as drop_collect
from edge.phase11.exp11a import run_windows, summarize as sum11a
from edge.phase11.footprint import collect as fp_collect, unit_spread
from edge.phase11.overlap import CountRule, evaluate
from edge.phase9.harness import make_mask
from scripts.phase5.data import window_path

OUT = Path("reports/phase11/final11.json")
if OUT.exists():
    raise SystemExit(f"{OUT} exists; the Phase 11 final set is evaluated once.")
man = json.loads(Path("reports/phase11/split_manifest_phase11.json").read_text())
wins = [dict(w) for w in man["windows"]]
rep = {"versions": versions(), "p11_hash": P11_HASH, "p11_params": P11_PARAMS, "n_windows": len(wins),
       "missing_dataset": [dict(session=w["session"], t0_s=w["t0_s"]) for w in wins if not window_path(w["pid"], w["t0_s"], w["duration_s"]).exists()]}
engines = ["amplitude", "template", "beamform", "weighted_scan", "robust_scan"]
t = time.time()
# 1. event referencing with P11 engines (identical to P10/P9 on intact contacts)
recs, ws = run_windows(wins, p11_factory, log=lambda s: print(s, flush=True))
rep["referencing"] = sum11a(recs, engines, ws)
rep["referencing_per_session"] = {s: sum11a([r for r in recs if r["session"] == s], engines, [w for w in ws if w.get("session") == s])
                                  for s in sorted({r["session"] for r in recs})}
pickle.dump({"recs": recs, "wstats": ws}, open("data/phase11/final11_referencing.pkl", "wb"))
print("referencing", round(time.time() - t), "s", flush=True)
# 2. dead-contact fallback
rows = drop_collect(build_events_shift(wins, 0.0), p9_factory, p10_factory)
rep["dropout"] = {k: apply(rows, *k.split("|")[:1], int(k.split("|")[1])) if False else apply(rows, k.split("|")[0], int(k.split("|")[1]))
                  for k in ("P9|0", "P10|0", "disagree>k->reject|80")}
rep["dropout_by_condition"] = {c: {k: apply([r for r in rows if r["cond"] == c], k.split("|")[0], int(k.split("|")[1]))
                                   for k in ("P9|0", "P10|0", "disagree>k->reject|80")} for c in sorted({r["cond"] for r in rows})}
rep["dropout_by_session"] = {s: {k: apply([r for r in rows if r["session"] == s], k.split("|")[0], int(k.split("|")[1]))
                                 for k in ("P9|0", "P10|0", "disagree>k->reject|80")} for s in sorted({r["session"] for r in rows})}
print("dropout", round(time.time() - t), "s", flush=True)
# 3. forward model check (line20 retained; linefree reported)
fev = fp_collect(wins, per_window=150, keep_low=True)
fits = {n: [F.fit(F.Model(n), e["a"], e["sigma"], e["E"], cache=c) for e in fev] for n, c in (("line20", {}), ("linefree", {}))}
sp = {n: unit_spread(fev, f) for n, f in fits.items()}
units = sorted(set(sp["line20"]) & set(sp["linefree"]))
a = np.array([sp["line20"][u] for u in units]); b = np.array([sp["linefree"][u] for u in units])
rep["forward_model"] = {"n_units": len(units), "line20_spread": float(np.median(a)), "linefree_spread": float(np.median(b)),
                        "rel": float(np.median(b) / np.median(a) - 1)}
print("forward", round(time.time() - t), "s", flush=True)
# 4. synthetic overlap + calibration on the fresh p11final pool
array, fin = make_stress("p11final", n_per_cell=20)
loc = localizer(array)
cal = json.loads(Path("reports/phase11/calibration_p11.json").read_text())
cmodel = {"table": {tuple(k.split("|")): v for k, v in cal["table"].items()}, "stratum": cal["stratum"], "all": cal["all"], "keys": tuple(cal["keys"])}
ov = {}
crow = []
rng = np.random.default_rng(20271401)
for name, rule in (("P10", CountRule()), ("P11_OV-A", CountRule(axial_only_boundary=True)),
                   ("experimental_K3", None)):
    if rule is None:
        continue
    R = {f: [evaluate(loc, e, rule) for e in evs] for f, evs in fin.items()}
    pairs = R["dt"] + R["sep"]; res = [r for r in pairs if r["res"]]
    ov[name] = {"resolvable_pairs": summarize(res, 2), "all_pairs": summarize(pairs, 2), "singles": summarize(R["single"], 1),
                "false_two_singles": float(np.mean([r["n_est"] >= 2 for r in R["single"]])),
                "simultaneous_resolvable": summarize([r for r in res if r["dt"] == 0], 2), "triples": summarize(R["triple"], 3)}
rep["overlap_p11final"] = ov
for fam, evs in fin.items():
    for e in evs:
        for drop in ("none", "dead"):
            zero = None if drop == "none" else ~make_mask(array.n_channels, 0.75, "random", rng)
            r = evaluate(loc, e, CountRule(axial_only_boundary=True), zero=zero)
            if r["primary_xy"] is None or not np.isfinite(r["d2"]) or not r["accepted"]:
                crow.append({"accepted": False}); continue
            truth = np.array([s[:2] for s in e.sources]); p_ = np.array(r["primary_xy"])
            k = int(np.argmin(np.linalg.norm(truth - p_, axis=1)))
            crow.append({"accepted": True, "truth_idx": k, "d2": r["d2"], "det": r["det"], "snr_g": CAL.snr_group(r["snr"]),
                         "stratum": CAL.stratum(r["flags"], r["n_est"]), "drop": drop, "err": float(np.linalg.norm(truth[k] - p_))})
S = [r for r in crow if r["accepted"] and r["truth_idx"] == 0]
rep["calibration_p11final"] = {"accepted_fraction": float(np.mean([r["accepted"] for r in crow])),
                               "scored_fraction_of_accepted": len(S) / max(sum(r["accepted"] for r in crow), 1),
                               "mondrian": CAL.evaluate(cmodel, S)}
rep["runtime_s"] = time.time() - t
OUT.write_text(json.dumps(rep, indent=1, default=float) + "\n")
print("written", OUT)
