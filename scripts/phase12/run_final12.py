"""Phase 12 FINAL evaluation - single use. Runs the frozen P12 protocol (edge/phase12/configs.py) on the fresh final
sessions Q-S (reports/phase12/split_manifest_phase12.json) and the p12final synthetic pool (fresh seed, fresh MEArec
template set data/phase12/biophys_final, noise from Q-S), then aggregates reports/phase12/final12.json.
Refuses to run if final12.json exists. No parameter is changed after this point."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase12.configs import P12_HASH, P12_PARAMS

OUT = Path("reports/phase12/final12.json")
if OUT.exists():
    raise SystemExit("final12.json exists: the Phase 12 final evaluation is single-use")
steps = [["scripts/phase12/run_12_2.py", "final", "--population", "matched"],
         ["scripts/phase12/run_12_2.py", "final", "--population", "all"],
         ["scripts/phase12/run_12_2b.py", "final", "0.9"],
         ["scripts/phase12/run_12_3.py", "final"],
         ["scripts/phase12/run_12_3_biophys.py", "final"],
         ["scripts/phase12/run_12_5_real.py", "final"],
         ["scripts/phase12/run_12_4.py", "build", "p12final"],
         ["scripts/phase12/run_12_4.py", "eval", "p12final"],
         ["scripts/phase12/run_12_5.py", "p12final"],
         ["scripts/phase12/run_12_4_real.py", "final"],
         ["scripts/phase12/run_12_6.py", "stream", "final"]]
t0 = time.time()
log = []
for s in steps:
    t = time.time()
    r = subprocess.run([sys.executable] + s, env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True)
    log.append({"step": " ".join(s), "returncode": r.returncode, "s": time.time() - t, "tail": r.stdout[-1500:] + r.stderr[-1500:]})
    print(log[-1]["step"], r.returncode, f"{time.time() - t:.0f}s", flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:]); break
R = Path("reports/phase12")
J = lambda p: json.loads((R / p).read_text()) if (R / p).exists() else None
OUT.write_text(json.dumps({"versions": versions(), "p12_hash": P12_HASH, "p12_params": P12_PARAMS, "steps": log,
                           "pooling_matched": J("exp12_2_final_matched.json"), "pooling_all": J("exp12_2_final_all.json"),
                           "online_groups": J("exp12_2b_final_rho0.9.json"), "single_event_real": J("exp12_3_final.json"),
                           "single_event_biophys": J("exp12_3_biophys_final.json"), "uncertainty_real_proxy": J("exp12_5_real_proxy_final.json"),
                           "overlap_p12final": J("exp12_4_p12final.json"), "uncertainty_p12final": J("exp12_5_uncertainty_p12final.json"),
                           "count_real_proxy": J("exp12_4_real_final.json"), "runtime_stream": J("exp12_6_stream_final.json"),
                           "runtime_s": time.time() - t0}, indent=1, default=float) + "\n")
print("final12 written", f"{time.time() - t0:.0f}s")
