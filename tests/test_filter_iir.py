import unittest

import numpy as np

from scripts.filter_iir import (
    IirConfig,
    design_sos,
    filter_adc_codes,
    maximum_pole_magnitude,
    response_at,
)


class FilterIirTest(unittest.TestCase):
    def setUp(self):
        self.config = IirConfig()
        self.sos = design_sos(self.config)

    def test_every_float_pole_is_inside_unit_circle(self):
        self.assertLess(maximum_pole_magnitude(self.sos), 1.0)

    def test_expected_frequency_regions(self):
        gains = response_at(
            self.sos,
            np.array([100.0, 300.0, 1_000.0, 6_000.0, 10_000.0]),
            self.config.sample_rate_hz,
        )
        self.assertLess(gains[0], -15.0)
        self.assertAlmostEqual(gains[1], -3.0, delta=0.2)
        self.assertGreater(gains[2], -0.5)
        self.assertAlmostEqual(gains[3], -3.0, delta=0.2)
        self.assertLess(gains[4], -8.0)

    def test_multichannel_output_is_finite_and_same_shape(self):
        impulse = np.zeros((4096, 3), dtype=np.int16)
        impulse[0] = [1000, -1000, 500]
        result = filter_adc_codes(impulse, 0.004119873046875, self.sos)
        self.assertEqual(result.shape, impulse.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(float(np.max(np.abs(result[-100:]))), 1e-12)


if __name__ == "__main__":
    unittest.main()
