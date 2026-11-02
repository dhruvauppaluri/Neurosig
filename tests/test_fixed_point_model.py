import unittest

import numpy as np
from scipy import signal

from scripts.detect_spikes import DetectorConfig
from scripts.filter_iir import IirConfig, design_sos, maximum_pole_magnitude
from scripts.fixed_point_model import (
    FixedPointConfig,
    dequantize_sos,
    detect_integer_candidates,
    fixed_iir_filter,
    merge_integer_events,
    quantize_sos,
)


class FixedPointModelTest(unittest.TestCase):
    def setUp(self):
        self.fixed_config = FixedPointConfig()
        self.float_sos = design_sos(IirConfig())
        self.quantized_sos = quantize_sos(self.float_sos, self.fixed_config)

    def test_quantized_coefficients_remain_stable(self):
        dequantized = dequantize_sos(self.quantized_sos, self.fixed_config)
        self.assertLess(maximum_pole_magnitude(dequantized), 1.0)

    def test_fixed_filter_tracks_float_and_does_not_saturate(self):
        n = np.arange(4096)
        input_codes = np.rint(2000 * np.sin(2 * np.pi * 1000 * n / 32000)).astype(np.int16)[:, None]
        fixed, stats = fixed_iir_filter(input_codes, self.quantized_sos, self.fixed_config)
        floating = signal.sosfilt(self.float_sos, input_codes.astype(float), axis=0)
        self.assertTrue(all(section["saturated_samples"] == 0 for section in stats))
        self.assertLess(float(np.sqrt(np.mean((fixed - floating) ** 2))), 15.0)

    def test_zero_input_after_impulse_settles_without_limit_cycle(self):
        impulse = np.zeros((20_000, 1), dtype=np.int16)
        impulse[0, 0] = 20_000
        fixed, _ = fixed_iir_filter(impulse, self.quantized_sos, self.fixed_config)
        np.testing.assert_array_equal(fixed[-1000:], 0)

    def test_integer_detector_merges_channels(self):
        config = DetectorConfig(sample_rate_hz=10_000, peak_search_ms=0.5, refractory_ms=1.0, merge_ms=0.5)
        samples = np.zeros((100, 2), dtype=np.int32)
        samples[20, :] = [-100, -120]
        thresholds = np.array([-50, -50], dtype=np.int32)
        merged = merge_integer_events(detect_integer_candidates(samples, thresholds, config), config)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].sample, merged[0].channel), (20, 1))


if __name__ == "__main__":
    unittest.main()
