from __future__ import annotations

import unittest

import numpy as np

from edge import SAMPLE_RATE_HZ, SEED
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.localize.overlap import spatial_overlap_conflict
from edge.localize.robust import RobustScanLocalizer
from edge.snippets import extract_snippet
from edge.types import snippet_sample_counts


class OverlapFailClosedTests(unittest.TestCase):
    def test_single_source_is_not_a_conflict(self) -> None:
        array = linear_probe_um()
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        t_peak = n_pre + 30
        voltages = render_recording(
            [PointSource(t_peak, (30.0, 300.0, 0.0), 12.0)],
            array, SAMPLE_RATE_HZ, noise_uv=0.2, seed=SEED, n_samples=t_peak + n_post + 10,
        )
        snippet = extract_snippet(voltages, t_peak, array, ground_truth_xyz_um=(30.0, 300.0, 0.0))
        conflict, _reason = spatial_overlap_conflict(snippet)
        self.assertFalse(conflict)
        estimate = RobustScanLocalizer(array=array).estimate(snippet)
        self.assertTrue(estimate.available)

    def test_two_separated_sources_fail_closed(self) -> None:
        array = linear_probe_um()
        n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
        t_peak = n_pre + 30
        sources = [
            PointSource(t_peak, (30.0, 200.0, 0.0), 12.0),
            PointSource(t_peak, (30.0, 420.0, 0.0), 12.0),
        ]
        voltages = render_recording(
            sources, array, SAMPLE_RATE_HZ, noise_uv=0.2, seed=SEED, n_samples=t_peak + n_post + 10,
        )
        snippet = extract_snippet(voltages, t_peak, array, ground_truth_xyz_um=(30.0, 200.0, 0.0))
        conflict, reason = spatial_overlap_conflict(snippet)
        self.assertTrue(conflict)
        self.assertIn("bimodal", reason)
        estimate = RobustScanLocalizer(array=array).estimate(snippet)
        self.assertFalse(estimate.available)
        self.assertIsNone(estimate.xyz_um)


if __name__ == "__main__":
    unittest.main()
