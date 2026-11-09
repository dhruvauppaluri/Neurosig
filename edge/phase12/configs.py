"""Frozen Phase 12 configuration P12 (decided on development data; checked on the selection split; frozen before the
final sessions Q-S and the p12final pool were evaluated)."""
from __future__ import annotations

import json
from pathlib import Path

from edge.phase8.registry import config_hash
from edge.phase11.configs import P11_HASH

ROOT = Path(__file__).resolve().parents[2]

P12_PARAMS = {
    "base": f"P11 {P11_HASH}",
    "single_event": {"alignment": "E3 (raw - median CAR, +-3 samples, contacts within 60 um)", "amplitude": "mean3",
                     "forward_model": "line20", "fit": "FastLine20 (4 um grid, 5 trust-region Gauss-Newton steps)",
                     "uncertainty": "disc of area c(snr_group) * identifiability area (exp12_5_real_proxy.json, dev)"},
    "unit_level": {"estimator": "l20_robust (Huber mean of E3 mean-trough amplitudes, line20)", "pool_sizes": [5, 10, 25, 50],
                   "online_grouping": {"rho": 0.9, "radius_um": 40, "min_group": 10}},
    "count": {"analysis": "Phase 9/10 K<=2 space-time fit with analytic-Jacobian refine and closed-form shared eigenvalue",
              "rule": "P12-LR (reports/phase12/count_model_p12.json)", "gate": "none (cheap gate rejected, 12.6)",
              "k3": "off (experimental)"},
    "region_multi": "Mondrian count x snr x chi1 tercile x dropout over all events (reports/phase12/calibration_p12.json)",
}
P12_HASH = config_hash(P12_PARAMS)


def count_model():
    return json.loads((ROOT / "reports/phase12/count_model_p12.json").read_text())
