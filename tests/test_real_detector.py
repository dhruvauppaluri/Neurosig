import unittest

import numpy as np

from scripts.detect_spikes import DetectorConfig, detect_candidates, estimate_noise, merge_channel_events
from scripts.real_detector import (
    FROZEN_REAL_DETECTOR_NAME,
    LEGACY_SIX_SIGMA,
    RealDetectorConfig,
    common_average_reference,
    detect_real,
    estimate_noise_robust,
    frozen_real_detector,
    reject_spatial_artifacts,
    sorter_aligned_snr,
)


class RealDetectorTests(unittest.TestCase):
    def test_six_sigma_misses_mid_amplitude_spikes_that_four_sigma_finds(self) -> None:
        samples = np.zeros((400, 1))
        samples[80, 0] = -4.5
        samples[220, 0] = -4.6
        baseline = np.zeros(1)
        sigma = np.ones(1)
        six = DetectorConfig(sample_rate_hz=10_000, threshold_sigma=6.0)
        four = DetectorConfig(sample_rate_hz=10_000, threshold_sigma=4.0)
        self.assertEqual(len(detect_candidates(samples, baseline, sigma, six)), 0)
        self.assertEqual(len(detect_candidates(samples, baseline, sigma, four)), 2)

    def test_car_recovers_spike_buried_in_common_mode(self) -> None:
        rng = np.random.default_rng(11)
        n_samples, n_channels = 2000, 8
        common = rng.normal(0.0, 18.0, size=n_samples)
        traces = rng.normal(0.0, 1.0, size=(n_samples, n_channels)) + common[:, None]
        traces[400:404, 0] -= np.array([8.0, 18.0, 10.0, 4.0])
        config = RealDetectorConfig(
            sample_rate_hz=10_000,
            highpass_hz=100.0,
            lowpass_hz=3_000.0,
            use_car=False,
            threshold_sigma=4.0,
            artifact_channel_fraction=1.0,
            mad_iterations=1,
        )
        without = detect_real(traces, config)
        with_car = detect_real(traces, RealDetectorConfig(**{**config.__dict__, "use_car": True}))
        self.assertGreater(with_car["merged_detections"], without["merged_detections"])
        self.assertIn(0, {event.channel for event in with_car["events"]})

    def test_artifact_veto_drops_all_channel_transients(self) -> None:
        from scripts.detect_spikes import Event

        events = [Event(10, channel, -20.0, 8.0) for channel in range(8)]
        events.append(Event(80, 1, -12.0, 6.0))
        config = RealDetectorConfig(sample_rate_hz=10_000, artifact_channel_fraction=0.5)
        kept, rejected = reject_spatial_artifacts(events, 8, config)
        self.assertEqual(rejected, 8)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].sample, 80)

    def test_legacy_preset_matches_original_negative_detector_on_isolated_spike(self) -> None:
        samples = np.zeros((300, 2))
        samples[40, 1] = -20.0
        samples += 1e-6
        legacy = detect_real(
            samples,
            RealDetectorConfig(
                sample_rate_hz=10_000,
                highpass_hz=100.0,
                lowpass_hz=3_000.0,
                use_car=False,
                threshold_sigma=6.0,
                artifact_channel_fraction=1.0,
                mad_iterations=0,
            ),
        )
        baseline, sigma = estimate_noise(legacy["filtered_uv"])
        original = merge_channel_events(
            detect_candidates(legacy["filtered_uv"], baseline, sigma, DetectorConfig(sample_rate_hz=10_000)),
            DetectorConfig(sample_rate_hz=10_000),
        )
        self.assertEqual(len(legacy["events"]), len(original))
        self.assertEqual(legacy["events"][0].sample, original[0].sample)
        self.assertEqual(LEGACY_SIX_SIGMA.threshold_sigma, 6.0)
        self.assertFalse(LEGACY_SIX_SIGMA.use_car)

    def test_iterative_mad_is_less_inflated_by_transients(self) -> None:
        rng = np.random.default_rng(3)
        samples = rng.normal(0.0, 2.0, size=(8_000, 1))
        samples[::50, 0] = -80.0
        _, naive = estimate_noise(samples)
        _, robust = estimate_noise_robust(samples, clip_sigma=6.0, iterations=3)
        self.assertLess(float(robust[0]), float(naive[0]))

    def test_sorter_aligned_snr_reports_fraction_above_thresholds(self) -> None:
        filtered = np.zeros((1000, 2))
        filtered[100, 0] = -5.0
        filtered[200, 0] = -7.0
        report = sorter_aligned_snr(filtered, np.zeros(2), np.ones(2), np.array([0.010, 0.020]), 10_000)
        self.assertEqual(report["n"], 2)
        self.assertGreater(report["median_peak_sigma"], 4.0)
        self.assertEqual(report["fraction_above_6_sigma"], 0.5)

    def test_car_preserves_differential_spike(self) -> None:
        traces = np.ones((5, 4))
        traces[2, 0] = -9.0
        referenced = common_average_reference(traces, "median")
        self.assertLess(referenced[2, 0], referenced[2, 1])

    def test_frozen_contract_is_car_four_sigma_hardware_band(self) -> None:
        cfg = frozen_real_detector(30_000.0)
        self.assertEqual(FROZEN_REAL_DETECTOR_NAME, "car_4sigma_300_6000_negative")
        self.assertTrue(cfg.use_car)
        self.assertEqual(cfg.threshold_sigma, 4.0)
        self.assertEqual(cfg.highpass_hz, 300.0)
        self.assertEqual(cfg.lowpass_hz, 6_000.0)
        self.assertEqual(cfg.polarity, "negative")
        self.assertNotEqual(cfg.threshold_sigma, LEGACY_SIX_SIGMA.threshold_sigma)


if __name__ == "__main__":
    unittest.main()
