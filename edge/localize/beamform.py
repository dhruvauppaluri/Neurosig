"""Quasi-static amplitude-pattern scan. This is NOT RF delay-and-sum beamforming.

Extracellular delays are not used. Action-potential propagation and finite
conduction velocity are ignored. Each steering vector is the predicted
quasi-static AMPLITUDE pattern of a monopole at a 3D grid location. The score
is normalized matched-filter / cosine power between that pattern and the
recorded peak-amplitude vector.
"""

from __future__ import annotations

import time

import numpy as np

from edge.forward import LAMBDA_UM, predicted_amplitude_patterns
from edge.geometry import linear_probe_um
from edge.localize.common import (
    electrode_com,
    finish_estimate,
    fold_linear_probe_ambiguity,
    l2_normalize,
    snippet_amplitudes,
    softmax_interpolate_xyz,
)
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet


def default_scan_grid_um(
    n_channels: int = 32,
    pitch_um: float = 20.0,
) -> np.ndarray:
    y_max = (n_channels - 1) * pitch_um
    # 3D scan; x>=0 because rotation about the shank is unidentifiable.
    xs = np.linspace(0.0, 80.0, 17)
    ys = np.linspace(0.0, y_max, 63)
    zs = np.linspace(-40.0, 40.0, 9)
    return np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T.astype(np.float64)


class BeamformLocalizer:
    """3D grid scan with quasi-static amplitude steering vectors.

    Not RF delay-and-sum: no time delays, no phase, no speed-of-sound steering.
    """

    method = "beamform"

    def __init__(
        self,
        array: ElectrodeArray | None = None,
        lambda_um: float = LAMBDA_UM,
        grid_xyz_um: np.ndarray | None = None,
        interpolate_top_k: int = 8,
        model: str = "offset_reciprocal",
    ) -> None:
        self.array = array or linear_probe_um()
        self.lambda_um = float(lambda_um)
        self.model = str(model)
        self.interpolate_top_k = int(interpolate_top_k)
        self.grid_xyz_um = (
            np.asarray(grid_xyz_um, dtype=np.float64)
            if grid_xyz_um is not None
            else default_scan_grid_um(self.array.n_channels)
        )
        if self.grid_xyz_um.ndim != 2 or self.grid_xyz_um.shape[1] != 3:
            raise ValueError("grid_xyz_um must be [n_grid, 3]")
        self.steering = predicted_amplitude_patterns(
            self.grid_xyz_um, self.array.xyz_um(), lambda_um=self.lambda_um, model=self.model
        )
        self.steering_n = l2_normalize(self.steering, axis=1)

    def score_profile(self, snippet: Snippet, n_bins: int = 32) -> tuple[float, ...]:
        """Maximum normalized spatial score in each probe-depth bin."""
        observed = l2_normalize(snippet_amplitudes(snippet))
        scores = self.steering_n @ observed
        edges = np.linspace(float(np.min(self.grid_xyz_um[:, 1])),
                            float(np.max(self.grid_xyz_um[:, 1])), n_bins + 1)
        groups = np.clip(np.digitize(self.grid_xyz_um[:, 1], edges) - 1, 0, n_bins - 1)
        return tuple(float(np.max(scores[groups == i])) if np.any(groups == i) else 0.0
                     for i in range(n_bins))

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        t0 = time.perf_counter()
        amps = snippet_amplitudes(snippet)
        if float(np.max(amps)) <= 0.0:
            return finish_estimate(
                method=self.method,
                xyz_um=electrode_com(snippet),
                snippet=snippet,
                t_start=t0,
                confidence=0.0,
                reason="zero amplitude; electrode centroid",
            )
        observed_n = l2_normalize(amps)
        # Normalized matched-filter / cosine power. Not RF delay-and-sum.
        scores = self.steering_n @ observed_n
        xyz_raw, peak = softmax_interpolate_xyz(
            self.grid_xyz_um, scores, top_k=self.interpolate_top_k
        )
        xyz = fold_linear_probe_ambiguity(
            (float(xyz_raw[0]), float(xyz_raw[1]), float(xyz_raw[2])),
            snippet.electrode_xyz_um,
        )
        contrast = peak - float(np.median(scores))
        confidence = float(np.clip(0.5 * max(0.0, peak) + 0.5 * max(0.0, contrast), 0.0, 1.0))
        return finish_estimate(
            method=self.method,
            xyz_um=xyz,
            snippet=snippet,
            t_start=t0,
            confidence=confidence,
            reason="quasi-static amplitude matched filter with top-k interpolation (not RF delay-and-sum)",
        )
