"""Catalog of forward-model amplitude templates evaluated at held-out sources."""

from __future__ import annotations

import time

import numpy as np

from edge.forward import LAMBDA_UM, predicted_amplitude_patterns
from edge.geometry import linear_probe_um
from edge.localize.common import electrode_com, finish_estimate, fold_linear_probe_ambiguity, l2_normalize, snippet_amplitudes
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet


def default_training_xyz_um(
    n_channels: int = 32,
    pitch_um: float = 20.0,
) -> np.ndarray:
    """Coarse training catalog. Tests and benchmarks must hold out other sites."""
    xs = np.array([10.0, 20.0, 35.0, 50.0, 70.0])
    zs = np.array([-25.0, 0.0, 25.0])
    ys = np.arange(0.0, (n_channels - 1) * pitch_um + 1.0, 40.0)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    return np.asarray(grid, dtype=np.float64)


class TemplateLocalizer:
    """Compare the recorded peak-amplitude vector to training-set templates.

    Templates are noiseless quasi-static amplitude patterns at a finite set of
    TRAINING source positions. At test time the observed vector is matched with
    cosine similarity (equivalently, least-squares scale fit on L2-normalized
    templates). Estimates interpolate the top-k training sites. Callers MUST
    evaluate on HELD-OUT source locations that were not used to build templates.
    """

    method = "template"

    def __init__(
        self,
        array: ElectrodeArray | None = None,
        training_xyz_um: np.ndarray | None = None,
        lambda_um: float = LAMBDA_UM,
        top_k: int = 8,
    ) -> None:
        self.array = array or linear_probe_um()
        self.lambda_um = float(lambda_um)
        self.top_k = int(top_k)
        self.training_xyz_um = (
            np.asarray(training_xyz_um, dtype=np.float64)
            if training_xyz_um is not None
            else default_training_xyz_um(self.array.n_channels)
        )
        if self.training_xyz_um.ndim != 2 or self.training_xyz_um.shape[1] != 3:
            raise ValueError("training_xyz_um must be [n_src, 3]")
        self.templates = predicted_amplitude_patterns(
            self.training_xyz_um, self.array.xyz_um(), lambda_um=self.lambda_um
        )
        self.templates_n = l2_normalize(self.templates, axis=1)

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
        scores = self.templates_n @ observed_n
        k = min(self.top_k, scores.shape[0])
        top = np.argpartition(scores, -k)[-k:]
        top_scores = np.maximum(scores[top], 0.0)
        # Softmax over cosine scores interpolates held-out locations.
        logits = top_scores / 0.05
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / max(float(np.sum(weights)), 1e-18)
        xyz = weights @ self.training_xyz_um[top]
        xyz_um = fold_linear_probe_ambiguity(
            (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            snippet.electrode_xyz_um,
        )
        confidence = float(np.max(scores))
        return finish_estimate(
            method=self.method,
            xyz_um=xyz_um,
            snippet=snippet,
            t_start=t0,
            confidence=confidence,
            reason="held-out cosine / LS match to training amplitude templates",
        )
