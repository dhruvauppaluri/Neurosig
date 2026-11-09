"""Phase 14.3 identifiability limit: separate model mismatch from noise for the frozen count model (ms3, spatial
whitening). The same p14select events are rendered noise-free (noise_scale = 0) and at noise_scale 0.5 / 1 / 2.
For noise-free singles, the two-source energy gain (SSE1 - SSE2, in whitened noise units at the true SNR) is the
phantom evidence that one neuron's own complexity supplies; it is compared with the gain of noise-free resolvable
pairs and with the calibrated threshold expressed in the same units. Writes reports/phase14/exp14_3_identifiability.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase14 import bench14 as B
from edge.phase14.neuron_model import NoiseModel, SourceModel, prepare

pool = sys.argv[1] if len(sys.argv) > 1 else "p14select"
nm = NoiseModel(B.noise_bank(B.POOLS[pool]["noise"]))
out = {"pool": pool, "levels": {}}
cache = {}
for scale in (0.0, 0.5, 1.0, 2.0):
    ev = B.generate(pool, 40, 30, 0, noise_scale=scale)
    rows = []
    for e in ev:
        E = np.asarray(e.snippet.electrode_xyz_um, float)
        k = E.tobytes()
        if k not in cache:
            cache[k] = SourceModel(E, "ms3")
        f = cache[k].fit(prepare(e.snippet, noise=nm, temporal=False), 2)
        rows.append({"kind": e.kind, "res": e.resolvable, "snr": e.snr, "gain": f["sse1"] - f["sse2"], "F": f["F21"], "T": f["T"]})
    S = [r for r in rows if r["kind"] == "single"]; P = [r for r in rows if r["kind"] == "pair" and r["res"]]
    snr = np.array([r["snr"][0] for r in S]); g = np.array([r["gain"] for r in S]); gp = np.array([r["gain"] for r in P])
    lvl = {"single_gain_median": float(np.median(g)), "pair_res_gain_median": float(np.median(gp)),
           "single_gain_by_snr": {f"{lo}-{hi}": float(np.median(g[(snr >= lo) & (snr < hi)])) for lo, hi in ((3, 6), (6, 10), (10, 15), (15, 21)) if np.any((snr >= lo) & (snr < hi))},
           "frac_pairs_gain_above_single_p98": float(np.mean(gp > np.quantile(g, 0.98))),
           "auc_gain": float(np.mean(gp[:, None] > g[None]))}
    out["levels"][str(scale)] = lvl
    print(scale, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in lvl.items()}, flush=True)
Path("reports/phase14/exp14_3_identifiability.json").write_text(json.dumps(out, indent=1) + "\n")
