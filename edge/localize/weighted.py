"""Locally weighted normalized spatial amplitude match; no time-delay steering."""

from __future__ import annotations

import time
import numpy as np

from edge.localize.beamform import BeamformLocalizer
from edge.localize.common import finish_estimate, fold_linear_probe_ambiguity, l2_normalize, snippet_amplitudes
from edge.types import LocalizationEstimate, Snippet


class WeightedScanLocalizer(BeamformLocalizer):
    method = "weighted_scan"

    def score_profile(self, snippet: Snippet, n_bins: int = 32) -> tuple[float, ...]:
        amplitudes = snippet_amplitudes(snippet)
        peak = int(np.argmax(amplitudes))
        distance = np.linalg.norm(snippet.electrode_xyz_um - snippet.electrode_xyz_um[peak], axis=1)
        weights = 0.25 + 0.75 * np.exp(-0.5 * (distance / 120.0) ** 2)
        scores = l2_normalize(self.steering * weights[None, :], axis=1) @ l2_normalize(amplitudes * weights)
        edges = np.linspace(float(np.min(self.grid_xyz_um[:, 1])),
                            float(np.max(self.grid_xyz_um[:, 1])), n_bins + 1)
        groups = np.clip(np.digitize(self.grid_xyz_um[:, 1], edges) - 1, 0, n_bins - 1)
        return tuple(float(np.max(scores[groups == i])) if np.any(groups == i) else 0.0
                     for i in range(n_bins))

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        t0 = time.perf_counter()
        amplitudes = snippet_amplitudes(snippet)
        if not np.any(amplitudes > 0):
            return super().estimate(snippet)
        peak = int(np.argmax(amplitudes))
        distance = np.linalg.norm(snippet.electrode_xyz_um - snippet.electrode_xyz_um[peak], axis=1)
        weights = 0.25 + 0.75 * np.exp(-0.5 * (distance / 120.0) ** 2)
        observed = l2_normalize(amplitudes * weights)
        patterns = l2_normalize(self.steering * weights[None, :], axis=1)
        scores = patterns @ observed
        best = int(np.argmax(scores))
        xyz = fold_linear_probe_ambiguity(tuple(float(v) for v in self.grid_xyz_um[best]),
                                          snippet.electrode_xyz_um)
        return finish_estimate(method=self.method, xyz_um=xyz, snippet=snippet, t_start=t0,
                               confidence=float(np.clip(scores[best], 0.0, 1.0)),
                               reason="locally weighted normalized quasi-static amplitude match")
