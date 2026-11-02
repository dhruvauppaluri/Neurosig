import unittest

import numpy as np

from scripts.quantize_adc import AdcConfig, quantize_voltages


class QuantizeAdcTest(unittest.TestCase):
    def setUp(self):
        self.config = AdcConfig(bits=16, full_scale_uv=135.0)

    def test_rounds_halves_away_from_zero(self):
        lsb = self.config.lsb_uv
        codes, saturated = quantize_voltages(
            np.array([0.0, 0.5 * lsb, -0.5 * lsb, 1.49 * lsb, -1.49 * lsb]),
            self.config,
        )
        np.testing.assert_array_equal(codes, [0, 1, -1, 1, -1])
        self.assertFalse(saturated.any())

    def test_saturates_at_signed_limits(self):
        codes, saturated = quantize_voltages(
            np.array([-200.0, -135.0, 0.0, 135.0, 200.0]), self.config
        )
        np.testing.assert_array_equal(codes, [-32768, -32768, 0, 32767, 32767])
        np.testing.assert_array_equal(saturated, [True, False, False, True, True])

    def test_measured_minimum_rounds_to_expected_code(self):
        codes, saturated = quantize_voltages(np.array([-129.17799377441406]), self.config)
        self.assertEqual(int(codes[0]), -31355)
        self.assertFalse(bool(saturated[0]))


if __name__ == "__main__":
    unittest.main()
