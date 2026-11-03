from __future__ import annotations

import unittest

import numpy as np

from edge import SAMPLE_RATE_HZ, SEED
from edge.forward import LAMBDA_UM, PointSource, predicted_amplitude_pattern, render_recording, triphasic_template
from edge.geometry import linear_probe_um


class ForwardModelTests(unittest.TestCase):
    def test_triphasic_has_unit_negative_peak(self) -> None:
        kernel = triphasic_template(80)
        self.assertAlmostEqual(float(np.min(kernel)), -1.0, places=6)
        self.assertTrue(np.isfinite(kernel).all())

    def test_gain_units_are_micrometers(self) -> None:
        electrodes = linear_probe_um().xyz_um()
        near = predicted_amplitude_pattern((30.0, 300.0, 0.0), electrodes)
        far = predicted_amplitude_pattern((200.0, 300.0, 0.0), electrodes)
        self.assertGreater(float(np.max(near)), float(np.max(far)))
        # 1/(r_mm + 0.08) at r=0: 1/0.08 = 12.5
        on_site = predicted_amplitude_pattern((0.0, 0.0, 0.0), electrodes[:1])
        self.assertAlmostEqual(float(on_site[0]), 1.0 / (LAMBDA_UM / 1000.0), places=6)
        line = predicted_amplitude_pattern((0.0, 0.0, 0.0), electrodes[:1], model="line_source")
        self.assertAlmostEqual(float(line[0]), 1.0 / LAMBDA_UM, places=6)

    def test_superposition_and_shape(self) -> None:
        array = linear_probe_um()
        sources = [
            PointSource(t_sample=80, xyz_um=(30.0, 200.0, 0.0), amplitude_uv=12.0),
            PointSource(t_sample=80, xyz_um=(30.0, 400.0, 0.0), amplitude_uv=12.0),
        ]
        voltages = render_recording(sources, array, SAMPLE_RATE_HZ, noise_uv=0.0, seed=SEED, n_samples=200)
        self.assertEqual(voltages.shape, (200, 32))
        self.assertTrue(np.isfinite(voltages).all())
        one = render_recording(sources[:1], array, SAMPLE_RATE_HZ, 0.0, SEED, n_samples=200)
        two = render_recording(sources[1:], array, SAMPLE_RATE_HZ, 0.0, SEED, n_samples=200)
        np.testing.assert_allclose(voltages, one + two, atol=1e-9)

    def test_noise_changes_with_seed(self) -> None:
        array = linear_probe_um()
        sources = [PointSource(t_sample=50, xyz_um=(25.0, 160.0, 0.0))]
        a = render_recording(sources, array, SAMPLE_RATE_HZ, 5.0, seed=1, n_samples=120)
        b = render_recording(sources, array, SAMPLE_RATE_HZ, 5.0, seed=2, n_samples=120)
        self.assertFalse(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
