from __future__ import annotations

import unittest

import numpy as np

from scripts.detect_spikes import DetectorConfig
from scripts.quantize_adc import AdcConfig
from viz.sim.events import NeuralEvent
from viz.sim.golden_bridge import process_window, refuse_spike_timestamps_as_voltage
from viz.sim.localization import NotImplementedLocalizer, LocalizationRequest, default_probe_array
from viz.sim.recording_model import RecordingConfig, render_voltages


def _near_probe(channel: int = 10) -> tuple[float, float, float]:
    site = default_probe_array().sites[channel]
    return (site.xyz_mm[0] + 0.05, site.xyz_mm[1], site.xyz_mm[2])


class GoldenBridgeTests(unittest.TestCase):
    def test_overlapping_templates_are_finite(self) -> None:
        events = [
            NeuralEvent("a", 0.004, _near_probe(8), "visual_cortex", -180.0, "u0", "synthetic_source"),
            NeuralEvent("b", 0.0042, _near_probe(9), "visual_cortex", -160.0, "u1", "synthetic_source"),
        ]
        voltages, array = render_voltages(
            events,
            0.0,
            0.02,
            config=RecordingConfig(noise_uv=0.5, attenuation_mm=0.4, seed=1),
        )
        self.assertEqual(array.sites[0].channel_id, 0)
        self.assertEqual(voltages.shape[1], 32)
        self.assertTrue(np.isfinite(voltages).all())
        self.assertLess(float(voltages.min()), -10.0)

    def test_quantized_detection_matches_ground_truth(self) -> None:
        events = [
            NeuralEvent("gt0", 0.006, _near_probe(12), "visual_cortex", -220.0, "u0", "synthetic_source"),
            NeuralEvent("gt1", 0.012, _near_probe(14), "visual_cortex", -210.0, "u1", "synthetic_source"),
        ]
        result = process_window(
            events,
            0.0,
            0.03,
            recording=RecordingConfig(noise_uv=0.4, attenuation_mm=0.35, seed=2),
            adc=AdcConfig(),
            detector=DetectorConfig(threshold_sigma=4.0, match_tolerance_ms=1.0),
        )
        self.assertEqual(result.codes.dtype, np.int16)
        self.assertGreaterEqual(result.metrics["true_positives"], 1)
        self.assertTrue(any(event.kind == "detected_spike" for event in result.events))
        self.assertFalse(result.localization["available"])

    def test_refuses_sorted_spikes_as_adc(self) -> None:
        with self.assertRaises(ValueError):
            refuse_spike_timestamps_as_voltage("neuropixels_spike")

    def test_localizer_is_placeholder(self) -> None:
        result = NotImplementedLocalizer().estimate(
            LocalizationRequest(channel_ids=(0,), electrode_xyz_mm=((0.0, 0.0, 0.0),), event_ids=("x",))
        )
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "Localization not implemented")


if __name__ == "__main__":
    unittest.main()
