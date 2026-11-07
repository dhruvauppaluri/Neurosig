"""Frozen Phase 9 configurations (hashes recorded in reports)."""
from __future__ import annotations

import numpy as np

from edge.localize.beamform import default_scan_grid_um
from edge.phase8.registry import config_hash
from edge.phase9.factories import baseline_factory, calibrated_factory
from edge.phase9.improved import RobustScanV2, conditioned_factory

P9_CONDITIONER = dict(baseline=True, car=True, interp_dead=True)
P9_PARAMS = {"lam_line_um": 20.0, "lam_mono_um": 30.0, "gain_model": "line_source", "conditioner": P9_CONDITIONER,
             "robust_overlap_rule": {"type": "noise_aware", "min_snr": 6.0}}
P9_HASH = config_hash(P9_PARAMS)


def p9_factory(array):
    eng = dict(calibrated_factory(P9_PARAMS["lam_line_um"], P9_PARAMS["lam_mono_um"])(array))
    shift = np.array([0.0, float(array.xyz_um()[:, 1].min()), 0.0])
    eng["robust_scan"] = RobustScanV2(array=array, grid_xyz_um=default_scan_grid_um(array.n_channels) + shift,
                                      model="line_source", lambda_um=P9_PARAMS["lam_line_um"],
                                      min_snr=P9_PARAMS["robust_overlap_rule"]["min_snr"])
    return {k: __import__("edge.phase9.improved", fromlist=["ConditionedLocalizer"]).ConditionedLocalizer(e, **P9_CONDITIONER)
            for k, e in eng.items()}


def legacy_factory(array):
    return baseline_factory(array)


def corrected_baseline_factory(array):
    """Existing five engines (legacy gains) with baseline-subtracted snippets: the null-controlled reference."""
    return conditioned_factory(baseline_factory, baseline=True)(array)
