"""Single-shot Phase 10 evaluation of the FROZEN P9 configuration (hash 5ed8a69b97eb) and its comparators on the
fresh locked sessions I-L. Same battery as Phase 9 (real + null, dropout, noise) plus the Phase 10 segment / SNR /
peak-electrode tests. Refuses to rerun. Nothing is tuned from these results."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, P9_PARAMS, corrected_baseline_factory, legacy_factory, p9_factory
from edge.phase9.evalset import build_events_shift, evaluate
from edge.phase10.configs import P10_HASH, P10_PARAMS, p10_factory
from edge.phase10.realextra import run_extra
from scripts.phase5.data import window_path

OUT = Path("reports/phase10/final_locked10.json")
if OUT.exists():
    raise SystemExit(f"{OUT} exists; the Phase 10 locked sessions are evaluated once.")
wins = H.manifest_windows("locked10")
missing = [w for w in wins if not window_path(w["pid"], w["t0_s"], w["duration_s"]).exists()]
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
facs = {"legacy_raw": legacy_factory, "corrected_B": corrected_baseline_factory, "P9": p9_factory, "P10": p10_factory}
rep = {"versions": versions(), "p9_hash": P9_HASH, "p9_params": P9_PARAMS, "p10_hash": P10_HASH, "p10_params": P10_PARAMS,
       "n_windows": len(wins),
       "missing_dataset": [dict(session=w["session"], t0_s=w["t0_s"]) for w in missing],
       "per_session_events": {}, "battery": {}, "extra_real": {}, "extra_null": {}, "per_session": {}}
for ew in real:
    s = ew["window"]["session"]; rep["per_session_events"][s] = rep["per_session_events"].get(s, 0) + len(ew["snippets"])
for k, f in facs.items():
    t = time.time()
    rep["battery"][k] = evaluate(f, real, null)
    rep["extra_real"][k] = run_extra(real, f)
    rep["extra_null"][k] = run_extra(null, f, peak_drop=False)
    print(k, round(time.time() - t), "s", flush=True)
# per-session battery without perturbations (stability only), every session reported individually
for s in sorted({w["session"] for w in wins}):
    rs = [e for e in real if e["window"]["session"] == s]; ns = [e for e in null if e["window"]["session"] == s]
    if rs:
        rep["per_session"][s] = {k: evaluate(f, rs, ns, dropout=False, noise=False) for k, f in facs.items()}
OUT.write_text(json.dumps(rep, indent=1, default=float) + "\n")
print("written", OUT)
