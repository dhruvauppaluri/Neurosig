"""Phase 11A decomposition on dev windows: perturbation curves + observed deviation by SNR / overlap / matched status
(from the cached exp11a dev records). Writes reports/phase11/decompose_11a_dev.json."""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase9 import harness as H
from edge.phase10.configs import p10_factory
from edge.phase11.decompose import run
from edge.phase11.exp11a import _unit_dev

wins = H.manifest_windows("dev")
res = run(wins, p10_factory)
ovrows = res.pop("_overlap_rows")
recs = pickle.load(open("data/phase11/exp11a_dev_K6.pkl", "rb"))["recs"]
A = [r for r in recs if r["strategy"] == "A" and not r["null"]]
C = {(r["win"], r["pair"]): r for r in recs if r["strategy"] == "C" and not r["null"]}
_, dv = _unit_dev(A, "weighted_scan")
bins = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 100)]
s = np.array([r["snr"] for r, _ in dv]); d = np.array([x for _, x in dv]); m = np.array([r["matched"] for r, _ in dv])
res["observed_dev_by_snr_um"] = [{"snr_bin": b, "n": int(((s >= b[0]) & (s < b[1])).sum()),
                                  "median": float(np.median(d[(s >= b[0]) & (s < b[1])])),
                                  "p90": float(np.percentile(d[(s >= b[0]) & (s < b[1])], 90)),
                                  "matched_fraction": float(m[(s >= b[0]) & (s < b[1])].mean())} for b in bins]
res["observed_dev_matched_vs_unmatched"] = {"matched": {"n": int(m.sum()), "median": float(np.median(d[m])), "p90": float(np.percentile(d[m], 90))},
                                            "unmatched": {"n": int((~m).sum()), "median": float(np.median(d[~m])), "p90": float(np.percentile(d[~m], 90))}}
# timing: how often does local alignment move the sorter time, and does deviation fall where it moves a lot
shift = np.array([abs(C[(r["win"], r["pair"])]["sample"] - r["sample"]) for r, _ in dv])
res["observed_dev_by_alignment_shift"] = [{"abs_shift_samples": b, "n": int(((shift >= b[0]) & (shift <= b[1])).sum()),
                                          "median": float(np.median(d[(shift >= b[0]) & (shift <= b[1])]))} for b in ((0, 0), (1, 2), (3, 5), (6, 6))]
# overlap proxy vs deviation (match rows by order within session/unit is not possible; use SNR-matched comparison)
res["fraction_sorter_events_without_trough"] = float(np.mean(s < 4.0))
Path("reports/phase11/decompose_11a_dev.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
print(json.dumps({k: v for k, v in res.items()}, indent=0, default=float)[:4000])
