"""Phase 11A: A/B/C event-referencing comparison. Usage: run_11a.py <split> [--align-half K]
split: dev (Phase 5 calibration+selection, sessions A-C) | select (Phase 5 held-out + Phase 6 E-H + Phase 10 I-L,
all previously used, model selection only). Engines: frozen P10 (= P9 on intact data)."""
import argparse, json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9 import harness as H
from edge.phase10.configs import P10_HASH, p10_factory
from edge.phase11.exp11a import run_windows, summarize

ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--align-half", type=int, default=6)
ap.add_argument("--tag", default="")
ap.add_argument("--align-mode", default="filtered", choices=["filtered", "raw"])
a = ap.parse_args()
if a.split == "dev":
    wins = H.manifest_windows("dev")
elif a.split == "select":
    wins = H.manifest_windows("heldout5") + H.manifest_windows("locked") + H.manifest_windows("locked10")
else:
    raise SystemExit("final split is run only by scripts/phase11/run_final11.py")
t = time.time()
recs, ws = run_windows(wins, p10_factory, align_half=a.align_half, log=lambda s: print(s, flush=True), align_mode=a.align_mode)
engines = ["amplitude", "template", "beamform", "weighted_scan", "robust_scan"]
res = {"versions": versions(), "split": a.split, "engines_config": f"P10 {P10_HASH}", "align_half": a.align_half, "align_mode": a.align_mode,
       "summary": summarize(recs, engines, ws), "per_session": {}, "runtime_s": time.time() - t}
for s in sorted({r["session"] for r in recs}):
    res["per_session"][s] = summarize([r for r in recs if r["session"] == s], engines,
                                      [w for w in ws if w.get("session") == s])
tag = a.tag or (f"{a.split}_K{a.align_half}" + ("" if a.align_mode == "filtered" else "_raw"))
Path("reports/phase11").mkdir(parents=True, exist_ok=True)
Path(f"reports/phase11/exp11a_{tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
cache = Path("data/phase11"); cache.mkdir(parents=True, exist_ok=True)
pickle.dump({"recs": recs, "wstats": ws, "wins": wins}, open(cache / f"exp11a_{tag}.pkl", "wb"))
S = res["summary"]
print(json.dumps(S["events"], default=float))
for e in engines:
    m = S["matched"]["engines"][e]
    print(f"{e:14s} matched spread A {m['A']['spread_median']:.1f} B {m['B']['spread_median']:.1f} C {m['C']['spread_median']:.1f} | "
          f"p90 A {m['A']['dev_p90']:.1f} B {m['B']['dev_p90']:.1f} C {m['C']['dev_p90']:.1f} | null A {m['A']['null_spread_median']:.1f} "
          f"B {m['B']['null_spread_median']:.1f} C {m['C']['null_spread_median']:.1f} | B vs A {m['B']['vs_A']} C vs A {m['C']['vs_A']}")
for s, o in S["full_stream"].items():
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "engines"},
          {e: (round(x['spread_median'] or -1, 1), round(x['valid_fraction'], 3)) for e, x in o["engines"].items()})
