from __future__ import annotations

import unittest
import numpy as np

from edge.ibl import probe_geometry, select_visp_channels


class IblLoaderTests(unittest.TestCase):
    def test_selects_visual_contacts_and_preserves_coordinates(self) -> None:
        acronyms = np.array(["VISp2/3", "MOs", "VISp4", "VISl"])
        ids = select_visp_channels(acronyms, 2)
        self.assertEqual(ids.tolist(), [0, 2])
        coords = np.array([[10, 100], [20, 120], [30, 140], [40, 160]])
        array = probe_geometry(ids, coords)
        self.assertEqual(array.channel_ids(), (0, 2))
        self.assertEqual(array.xyz_um().tolist(), [[10.0, 100.0, 0.0], [30.0, 140.0, 0.0]])

    def test_falls_back_to_vis_when_visp_is_short(self) -> None:
        ids = select_visp_channels(np.array(["VISa", "MOs", "VISl", "CA1"]), 2)
        self.assertEqual(ids.tolist(), [0, 2])

    def test_rejects_too_few_visual_contacts(self) -> None:
        with self.assertRaises(ValueError):
            select_visp_channels(np.array(["VISp", "MOs"]), 2)


if __name__ == "__main__":
    unittest.main()
