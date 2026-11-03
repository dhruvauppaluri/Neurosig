from __future__ import annotations

import unittest

from viz.sim.localization import default_probe_array

from edge.coords import display_mm_to_probe_um, euclidean_um, probe_um_to_display_mm
from edge.geometry import linear_probe_um


class CoordTests(unittest.TestCase):
    def test_round_trip_origin(self) -> None:
        origin = (0.0, 0.0, 0.0)
        display = probe_um_to_display_mm(origin)
        self.assertEqual(display, (4.0, -80.0, 8.0))
        restored = display_mm_to_probe_um(display)
        self.assertTrue(euclidean_um(origin, restored) < 1e-9)

    def test_display_matches_viz_probe_contacts(self) -> None:
        scientific = linear_probe_um()
        visual = default_probe_array()
        for site, viz_site in zip(scientific.sites, visual.sites):
            mapped = probe_um_to_display_mm(site.xyz_um)
            self.assertAlmostEqual(mapped[0], viz_site.xyz_mm[0], places=6)
            self.assertAlmostEqual(mapped[1], viz_site.xyz_mm[1], places=6)
            self.assertAlmostEqual(mapped[2], viz_site.xyz_mm[2], places=6)


if __name__ == "__main__":
    unittest.main()
