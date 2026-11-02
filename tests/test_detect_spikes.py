import unittest

import numpy as np

from scripts.detect_spikes import (
    DetectorConfig,
    Event,
    detect_candidates,
    estimate_noise,
    match_events,
    merge_channel_events,
)


class DetectSpikesTest(unittest.TestCase):
    def setUp(self):
        self.config = DetectorConfig(sample_rate_hz=10_000, peak_search_ms=0.5, refractory_ms=1.0, merge_ms=0.5)

    def test_robust_noise_estimate_ignores_large_outlier(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(0, 2.0, size=(20_000, 2))
        samples[100, 0] = -1000
        baseline, sigma = estimate_noise(samples)
        np.testing.assert_allclose(baseline, 0, atol=0.08)
        np.testing.assert_allclose(sigma, 2.0, rtol=0.03)

    def test_refractory_and_cross_channel_merge(self):
        samples = np.zeros((100, 2))
        samples[20, :] = [-8, -10]
        samples[25, :] = [-9, -11]
        baseline = np.zeros(2)
        sigma = np.ones(2)
        candidates = detect_candidates(samples, baseline, sigma, self.config)
        merged = merge_channel_events(candidates, self.config)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].sample, 25)
        self.assertEqual(merged[0].channel, 1)

    def test_ordered_event_matching_counts_errors(self):
        detections = [Event(100, 0, -10, 10), Event(300, 0, -10, 10), Event(500, 0, -10, 10)]
        truth_seconds = np.array([0.0101, 0.0502])
        _, metrics = match_events(detections, truth_seconds, self.config)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)


if __name__ == "__main__":
    unittest.main()
