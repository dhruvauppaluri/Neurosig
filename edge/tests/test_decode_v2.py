from __future__ import annotations

import unittest
import numpy as np

from edge.decode.experiment_v2 import classification
from edge.decode.features import y_occupancy
from edge.decode.model import mcnemar_pvalue
from edge.decode.spatial_task import generate_dataset
from edge.geometry import linear_probe_um


class DecodeV2Tests(unittest.TestCase):
    def test_no_ground_truth_fallback(self) -> None:
        from edge.localize.base import NotImplementedLocalizer
        dataset = generate_dataset(seed=12, examples=3, localizer=NotImplementedLocalizer())
        positions = linear_probe_um().xyz_um()
        for channels, estimates in zip(dataset.spike_channels, dataset.estimated_xyz):
            np.testing.assert_array_equal(estimates, positions[channels])

    def test_confusion_and_class_metrics(self) -> None:
        metrics = classification(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_occupancy_uses_requested_y_range(self) -> None:
        occ = y_occupancy(np.array([2720.0, 2900.0]), n_bins=8, y_min=2700.0, y_max=3000.0)
        self.assertEqual(int(np.sum(occ)), 2)
        self.assertGreater(float(occ[0]), 0.0)
        self.assertGreater(float(occ[-1] + occ[-2] + occ[-3]), 0.0)

    def test_mcnemar_detects_identical_predictions(self) -> None:
        y = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 1, 1])
        result = mcnemar_pvalue(y, pred, pred)
        self.assertEqual(result["n_discordant"], 0)
        self.assertEqual(result["p_two_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
