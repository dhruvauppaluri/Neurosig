"""Development comparison: legacy raw | corrected baseline B | P9 frozen config. Real + null events, full noise battery."""
import json, sys, time
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, corrected_baseline_factory, legacy_factory, p9_factory
from edge.phase9.evalset import build_events_shift, evaluate, table
wins = H.manifest_windows("dev")
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
res = {k: evaluate(f, real, null) for k, f in {"legacy raw": legacy_factory, "corrected B": corrected_baseline_factory, "P9": p9_factory}.items()}
json.dump({"p9_hash": P9_HASH, "windows": len(wins), "results": res}, open("reports/phase9/dev_comparison.json", "w"), indent=1, default=float)
print(P9_HASH); print(table(res))
