"""Amplitude-weighted barycenter localizer with a 1/r lateral residual fit."""

from __future__ import annotations

import time

import numpy as np

from edge.forward import LAMBDA_UM, predicted_amplitude_patterns
from edge.localize.common import (
    electrode_com,
    finish_estimate,
    fold_linear_probe_ambiguity,
    l2_normalize,
    snippet_amplitudes,
)
from edge.types import LocalizationEstimate, Snippet


class AmplitudeLocalizer:
    """Barycenter of |peak| amplitudes on channels above a fraction of the max.

    Lateral (x, z) coordinates are not observable from a barycenter on a linear
    probe (all contacts have x=z=0). After estimating y from the thresholded
    amplitude COM, a 1/r residual is fit on the identifiable +x / z=0 plane
    with a short y refinement. Confidence mixes spatial concentration with the
    1/r cosine residual.
    """

    method = "amplitude"

    def __init__(
        self,
        amplitude_fraction: float = 0.5,
        lambda_um: float = LAMBDA_UM,
        lateral_max_um: float = 120.0,
        n_x: int = 49,
        n_y: int = 17,
    ) -> None:
        self.amplitude_fraction = float(amplitude_fraction)
        self.lambda_um = float(lambda_um)
        self.lateral_max_um = float(lateral_max_um)
        self.n_x = int(n_x)
        self.n_y = int(n_y)

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        t0 = time.perf_counter()
        amps = snippet_amplitudes(snippet)
        electrodes = np.asarray(snippet.electrode_xyz_um, dtype=np.float64)
        peak = float(np.max(amps))
        if peak <= 0.0:
            xyz = electrode_com(snippet)
            return finish_estimate(
                method=self.method,
                xyz_um=xyz,
                snippet=snippet,
                t_start=t0,
                confidence=0.0,
                reason="zero amplitude; electrode centroid",
            )

        mask = amps >= self.amplitude_fraction * peak
        if not np.any(mask):
            mask = amps >= peak
        weights = np.where(mask, amps, 0.0)
        _x_com, y_hat, _z_com = electrode_com(snippet, weights)

        # Search radial distance in the +x / z=0 plane about the amplitude COM y,
        # plus a short y refinement. Full 3D is unidentifiable on a linear probe.
        xs = np.linspace(0.0, self.lateral_max_um, self.n_x)
        ys = np.linspace(y_hat - 40.0, y_hat + 40.0, self.n_y)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        candidates = np.stack(
            [xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float64)],
            axis=1,
        )
        patterns = predicted_amplitude_patterns(candidates, electrodes, lambda_um=self.lambda_um)
        amps_n = l2_normalize(amps)
        patterns_n = l2_normalize(patterns, axis=1)
        scores = patterns_n @ amps_n
        best = int(np.argmax(scores))
        xyz = fold_linear_probe_ambiguity(
            (float(candidates[best, 0]), float(candidates[best, 1]), float(candidates[best, 2])),
            electrodes,
        )
        residual = float(scores[best])

        ys = electrodes[:, 1]
        wsum = float(np.sum(weights))
        spread = float(np.sqrt(np.sum(weights * (ys - y_hat) ** 2) / max(wsum, 1e-18)))
        pitch = float(np.median(np.diff(np.unique(np.round(ys, 6))))) if electrodes.shape[0] > 1 else 20.0
        if not np.isfinite(pitch) or pitch <= 0:
            pitch = 20.0
        concentration = 1.0 / (1.0 + spread / pitch)
        confidence = 0.5 * concentration + 0.5 * max(0.0, residual)
        return finish_estimate(
            method=self.method,
            xyz_um=xyz,
            snippet=snippet,
            t_start=t0,
            confidence=confidence,
            reason="amplitude barycenter + 1/r residual",
        )
