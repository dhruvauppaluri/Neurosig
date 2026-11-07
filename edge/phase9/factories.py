"""Engine factories used by Phase 9 experiments."""
from __future__ import annotations

import numpy as np

from edge.localize import get_localizer
from edge.localize.beamform import default_scan_grid_um
from edge.localize.template import default_training_xyz_um
from edge.types import ElectrodeArray


def baseline_factory(array: ElectrodeArray) -> dict:
    """Existing five engines with the edge.ibl_eval configuration (robust_scan: line_source)."""
    y_shift = float(np.min(array.xyz_um()[:, 1]))
    shift = np.array([0.0, y_shift, 0.0])
    training = default_training_xyz_um(array.n_channels) + shift
    scan = default_scan_grid_um(array.n_channels) + shift
    return {
        "amplitude": get_localizer("amplitude"),
        "template": get_localizer("template", array=array, training_xyz_um=training),
        "beamform": get_localizer("beamform", array=array, grid_xyz_um=scan),
        "weighted_scan": get_localizer("weighted_scan", array=array, grid_xyz_um=scan),
        "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan, model="line_source"),
    }


def calibrated_factory(lam_line: float = 20.0, lam_mono: float = 30.0, model: str = "line_source"):
    """Forward-model calibrated to the measured real footprint (reports/phase9/real_footprint.json)."""
    def f(array: ElectrodeArray) -> dict:
        shift = np.array([0.0, float(np.min(array.xyz_um()[:, 1])), 0.0])
        training = default_training_xyz_um(array.n_channels) + shift
        scan = default_scan_grid_um(array.n_channels) + shift
        return {
            "amplitude": get_localizer("amplitude", lambda_um=lam_mono),
            "template": get_localizer("template", array=array, training_xyz_um=training, lambda_um=lam_mono),
            "beamform": get_localizer("beamform", array=array, grid_xyz_um=scan, lambda_um=lam_line, model=model),
            "weighted_scan": get_localizer("weighted_scan", array=array, grid_xyz_um=scan, lambda_um=lam_line, model=model),
            "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan, lambda_um=lam_line, model=model),
        }
    return f
