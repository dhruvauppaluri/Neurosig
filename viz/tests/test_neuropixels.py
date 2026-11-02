from __future__ import annotations

import unittest

from viz.sim.neuropixels import DISCLAIMER, NeuropixelsPlayer, load_recording
from viz.sim.regions import NEUROPIXELS_ALLOWED_REGIONS, get_region, map_mouse_area


class NeuropixelsLoaderTests(unittest.TestCase):
    def test_fixture_fields(self) -> None:
        recording = load_recording()
        self.assertEqual(recording.species, "Mus musculus")
        self.assertFalse(recording.has_raw_voltage)
        self.assertGreater(len(recording.units), 3)
        self.assertGreater(len(recording.channels), 3)
        self.assertTrue(recording.stimulus)
        unit = recording.units[0]
        self.assertGreater(unit.spike_times_s.size, 0)
        self.assertIsNotNone(unit.peak_channel)

    def test_playback_times_and_regions(self) -> None:
        recording = load_recording()
        player = NeuropixelsPlayer(recording, loop=False)
        events = player.generate(0.0, 0.5)
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event.kind, "neuropixels_spike")
            self.assertIn(event.region_id, NEUROPIXELS_ALLOWED_REGIONS)
            self.assertTrue(get_region(event.region_id).contains(event.xyz_mm, margin=1.05))
            self.assertLess(event.t_s, 0.5)
            self.assertIn(event.position_quality, {"unit_ccf", "electrode_channel", "unknown"})
            self.assertTrue(event.metadata["not_raw_voltage"])

    def test_rejects_non_visual_mouse_area(self) -> None:
        with self.assertRaises(ValueError):
            map_mouse_area("CA1")

    def test_disclaimer_present(self) -> None:
        self.assertIn("mouse", DISCLAIMER.lower())


if __name__ == "__main__":
    unittest.main()
