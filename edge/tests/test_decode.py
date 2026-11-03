from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from edge.decode.experiment import NEUROPIXELS_ZIP, run_neuropixels_count_only, run_synthetic_comparison
from edge.decode.features import N_CHANNELS, SPATIAL_DIM, stack_baseline, stack_spatial
from edge.decode.model import train
from edge.decode.spatial_task import N_TEST, N_TRAIN, feature_matrices, generate_dataset, split_indices
from scripts.synthetic_decoder import train as train32


class DecodeFeatureTests(unittest.TestCase):
    def test_feature_shapes(self) -> None:
        dataset = generate_dataset(seed=20261103, examples=32)
        baseline, spatial = feature_matrices(dataset)
        self.assertEqual(baseline.shape, (32, N_CHANNELS))
        self.assertEqual(spatial.shape, (32, SPATIAL_DIM))
        self.assertTrue(np.array_equal(baseline, stack_baseline(dataset.counts)))
        self.assertEqual(stack_spatial(dataset.counts, dataset.estimated_xyz).shape[1], SPATIAL_DIM)
        self.assertTrue(dataset.used_stub_forward)
        self.assertEqual(dataset.counts.dtype, np.uint8)


class DecodeRegressionTests(unittest.TestCase):
    def test_both_models_run_on_seeded_split(self) -> None:
        result = run_synthetic_comparison(seed=20261103, examples=128, n_train=96)
        self.assertEqual(result["train_examples"], 96)
        self.assertEqual(result["test_examples"], 32)
        self.assertTrue(result["same_split"])
        self.assertEqual(result["baseline"]["n_features"], 32)
        self.assertEqual(result["baseline"]["trainer"], "scripts.synthetic_decoder.train")
        self.assertGreater(result["spatial"]["n_features"], 32)
        for key in ("baseline", "spatial"):
            acc = result[key]["test_accuracy"]
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
        delta = result["accuracy_delta_spatial_minus_baseline"]
        self.assertEqual(result["spatial_improved"], delta > 0)
        self.assertEqual(
            result["spatial_improved"],
            result["spatial"]["test_accuracy"] > result["baseline"]["test_accuracy"],
        )

    def test_32d_train_matches_synthetic_decoder(self) -> None:
        rng = np.random.default_rng(0)
        counts = rng.integers(0, 20, size=(64, 32)).astype(np.float64)
        labels = rng.integers(0, 2, size=64).astype(np.float64)
        w_a, b_a = train(counts, labels)
        w_b, b_b = train32(counts, labels)
        np.testing.assert_allclose(w_a, w_b)
        self.assertEqual(b_a, b_b)

    def test_split_is_fixed(self) -> None:
        train_idx, test_idx = split_indices()
        self.assertEqual(len(train_idx), N_TRAIN)
        self.assertEqual(len(test_idx), N_TEST)
        self.assertTrue(np.array_equal(train_idx, np.arange(N_TRAIN)))


class NeuropixelsSkipTests(unittest.TestCase):
    def test_skip_when_zip_missing_or_record_count_only(self) -> None:
        result = run_neuropixels_count_only(NEUROPIXELS_ZIP)
        self.assertEqual(result["spatial_features"], "off")
        if NEUROPIXELS_ZIP.exists():
            self.assertEqual(result["status"], "ran_count_only")
            self.assertIn("float_test_accuracy", result)
        else:
            self.assertEqual(result["status"], "skipped")
            self.assertIn("not present", result["limitation"])
        self.assertFalse(Path("viz/fixtures/neuropixels_visual_coding_mini.json").read_text() == "")


if __name__ == "__main__":
    unittest.main()
