from __future__ import annotations

import unittest
import numpy as np

from edge.decode.experiment_v2 import classification
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


if __name__ == "__main__":
    unittest.main()
