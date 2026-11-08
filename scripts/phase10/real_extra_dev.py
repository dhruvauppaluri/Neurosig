"""Phase 10 development run: segment, SNR and peak-electrode tests on dev windows (sessions A-D) plus the null set.
Compares legacy raw, corrected baseline B and frozen P9 (hash 5ed8a69b97eb). No parameters are tuned here."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from edge.phase8.registry import versions
from edge.phase9 import harness as H
from edge.phase9.configs import P9_HASH, corrected_baseline_factory, legacy_factory, p9_factory
from edge.phase9.evalset import build_events_shift
from edge.phase10.realextra import run_extra

kind = sys.argv[1] if len(sys.argv) > 1 else "dev"
out_path = Path(f"reports/phase10/real_extra_{kind}.json")
wins = H.manifest_windows(kind)
real, null = build_events_shift(wins, 0.0), build_events_shift(wins, 0.020)
facs = {"legacy_raw": legacy_factory, "corrected_B": corrected_baseline_factory, "P9": p9_factory}
res = {"versions": versions(), "p9_hash": P9_HASH, "windows": len(wins), "split": kind, "real": {}, "null": {}}
for k, f in facs.items():
    t = time.time()
    res["real"][k] = run_extra(real, f)
    res["null"][k] = run_extra(null, f, peak_drop=False)
    print(k, round(time.time() - t), "s", flush=True)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(res, indent=1, default=float) + "\n")
for k in facs:
    for m, x in res["real"][k].items():
        n = res["null"][k][m]
        print(f"{k:12s} {m:15s} valid {x['valid_fraction']:.2f} spread {x['within_unit_spread_um'].get('median', float('nan')):5.1f} "
              f"(null {n['within_unit_spread_um'].get('median', float('nan')):5.1f}) segShift {x['segment_shift_um'].get('median', float('nan')):5.1f} "
              f"(null {n['segment_shift_um'].get('median', float('nan')):5.1f}) rhoSNR {x.get('deviation_snr_spearman', float('nan')):+.2f} "
              f"peakK {x['drop_peak_known'].get('median', float('nan')):5.1f} peakU {x['drop_peak_unknown'].get('median', float('nan')):5.1f}")
