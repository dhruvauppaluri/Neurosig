"""Phase 10: the frozen Phase 9 decoding protocol on the four NEW insertions I-L (never used for decoding), plus an
MLP shuffled-position control so that the spatial MLP is compared with an identical-capacity model whose positions
carry no unit identity. Nothing is tuned; folds, models and hyperparameters are those of Phase 8/9.
Run: PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase10_new_sessions
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from edge.decode.phase8_ablation import cv_predictions, load_session, paired_boot, pick_units, spatial_features, trial_table
from edge.decode.phase9_ablation import p9_unit_locations, run_session
from edge.phase9.configs import P9_HASH

ROOT = Path(__file__).resolve().parents[2]


def mlp_shuffle_control(pid, seed=20269400):
    s = load_session(pid)
    units, _ = pick_units(s)
    counts, labels = trial_table(s, units)
    p9, p9u = p9_unit_locations(s, units)
    ymin, ymax = float(p9[:, 1].min() - 20), float(p9[:, 1].max() + 20)
    perm = np.random.default_rng(seed).permutation(len(units))
    F = {"mlp_p9+unc": spatial_features(counts, p9, ymin, ymax, extra=p9u),
         "mlp_p9_shuffled+unc": spatial_features(counts, p9[perm], ymin, ymax, extra=p9u[perm])}
    corr = cv_predictions(F, labels, seed)
    return {"accuracy": {k: float(v.mean()) for k, v in corr.items()},
            "mlp_p9_vs_mlp_shuffled": paired_boot(corr["mlp_p9_shuffled+unc"], corr["mlp_p9+unc"], seed + 99)}


def main():
    man = json.loads((ROOT / "reports/phase10/split_manifest_phase10.json").read_text())
    out = []
    for lab, info in man["sessions"].items():
        t = time.time()
        try:
            r = run_session(lab, info["pid"])
            if r.get("status") == "ran":
                r["mlp_capacity_control"] = mlp_shuffle_control(info["pid"])
            out.append(r)
        except Exception as exc:  # noqa: BLE001
            out.append({"session": lab, "pid": info["pid"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(lab, out[-1].get("status"), out[-1].get("accuracy") or out[-1].get("error") or out[-1].get("reason"),
              f"{time.time()-t:.0f}s", flush=True)
        (ROOT / "reports/phase10/decode_new_sessions.json").write_text(
            json.dumps({"p9_hash": P9_HASH, "protocol": "Phase 9 decode protocol, frozen", "sessions": out}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
