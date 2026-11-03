"""Noise-aware quasi-static scan with local refinement. Still not RF beamforming.

Improvements over the discrete cosine scan:
- aligned trough amplitudes (one time sample, not independent channel mins)
- channel weights from snippet-edge noise and distance to the peak contact
- top-k interpolation plus a fine local grid
- optional line-source decay for real probes; default matches the synthetic monopole

Overlapping sources are handled by spatially windowing around the strongest
contact. A second source is not unmixed; residual energy is reported as
confidence loss rather than a second coordinate.
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


def _edge_noise(snippet: Snippet) -> np.ndarray:
    voltages = np.asarray(snippet.voltages_uv, dtype=np.float64)
    n = voltages.shape[0]
    edge = max(4, n // 5)
    edges = np.concatenate([voltages[:edge], voltages[-edge:]], axis=0)
    noise = np.std(edges, axis=0)
    return np.maximum(noise, 1e-6)


def _local_grid(center: np.ndarray, span_um: tuple[float, float, float], steps: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(center[i] - span_um[i], center[i] + span_um[i], steps[i]) for i in range(3)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T.astype(np.float64)


class RobustScanLocalizer:
    """Adaptive, noise-weighted quasi-static amplitude scan."""

    method = "robust_scan"

    def __init__(
        self,
        array: ElectrodeArray | None = None,
        lambda_um: float = LAMBDA_UM,
        grid_xyz_um: np.ndarray | None = None,
        model: str = "offset_reciprocal",
        spatial_radius_um: float = 80.0,
        interpolate_top_k: int = 8,
    ) -> None:
        self.array = array or linear_probe_um()
        self.lambda_um = float(lambda_um)
        self.model = str(model)
        self.spatial_radius_um = float(spatial_radius_um)
        self.interpolate_top_k = int(interpolate_top_k)
        from edge.localize.beamform import default_scan_grid_um

        self.grid_xyz_um = (
            np.asarray(grid_xyz_um, dtype=np.float64)
            if grid_xyz_um is not None
            else default_scan_grid_um(self.array.n_channels)
        )
        self.steering = predicted_amplitude_patterns(
            self.grid_xyz_um, self.array.xyz_um(), lambda_um=self.lambda_um, model=self.model
        )

    def score_profile(self, snippet: Snippet, n_bins: int = 32) -> tuple[float, ...]:
        observed, weights = self._observed(snippet)
        patterns = l2_normalize(self.steering * weights[None, :], axis=1)
        scores = patterns @ observed
        edges = np.linspace(float(np.min(self.grid_xyz_um[:, 1])),
                            float(np.max(self.grid_xyz_um[:, 1])), n_bins + 1)
        groups = np.clip(np.digitize(self.grid_xyz_um[:, 1], edges) - 1, 0, n_bins - 1)
        return tuple(float(np.max(scores[groups == i])) if np.any(groups == i) else 0.0
                     for i in range(n_bins))

    def _observed(self, snippet: Snippet) -> tuple[np.ndarray, np.ndarray]:
        amps = snippet_amplitudes(snippet)
        noise = _edge_noise(snippet)
        peak = int(np.argmax(amps))
        distance = np.linalg.norm(snippet.electrode_xyz_um - snippet.electrode_xyz_um[peak], axis=1)
        spatial = np.exp(-0.5 * (distance / max(self.spatial_radius_um, 1.0)) ** 2)
        noise_w = 1.0 / np.square(noise)
        weights = 0.15 + 0.85 * spatial * (noise_w / max(float(np.max(noise_w)), 1e-18))
        return l2_normalize(amps * weights), weights

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
        observed, weights = self._observed(snippet)
        patterns = l2_normalize(self.steering * weights[None, :], axis=1)
        scores = patterns @ observed
        coarse, peak = softmax_interpolate_xyz(self.grid_xyz_um, scores, top_k=self.interpolate_top_k)
        local = _local_grid(coarse, (18.0, 36.0, 18.0), (9, 19, 7))
        local_steering = predicted_amplitude_patterns(
            local, snippet.electrode_xyz_um, lambda_um=self.lambda_um, model=self.model
        )
        local_patterns = l2_normalize(local_steering * weights[None, :], axis=1)
        local_scores = local_patterns @ observed
        xyz_raw, local_peak = softmax_interpolate_xyz(local, local_scores, top_k=self.interpolate_top_k)
        if local_peak + 1e-9 < peak:
            xyz_raw, local_peak = coarse, peak
        xyz = fold_linear_probe_ambiguity(tuple(float(v) for v in xyz_raw), snippet.electrode_xyz_um)
        residual = float(np.linalg.norm(observed - local_patterns[int(np.argmax(local_scores))]))
        confidence = float(np.clip(0.6 * max(0.0, local_peak) + 0.4 * max(0.0, 1.0 - residual), 0.0, 1.0))
        return finish_estimate(
            method=self.method,
            xyz_um=xyz,
            snippet=snippet,
            t_start=t0,
            confidence=confidence,
            reason="noise-weighted quasi-static scan with local interpolation (not RF delay-and-sum)",
        )
