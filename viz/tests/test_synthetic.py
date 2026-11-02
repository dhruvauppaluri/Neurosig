from __future__ import annotations

import unittest

from viz.sim.engine import NeuralEventEngine
from viz.sim.regions import get_region
from viz.sim.synthetic import SyntheticGenerator


class SyntheticEngineTests(unittest.TestCase):
    def test_same_seed_reproduces_events(self) -> None:
        first = SyntheticGenerator(seed=42)
        second = SyntheticGenerator(seed=42)
        a = first.generate(0.0, 0.4)
        b = second.generate(0.0, 0.4)
        self.assertGreater(len(a), 5)
        self.assertEqual([event.event_id for event in a], [event.event_id for event in b])
        self.assertEqual([event.t_s for event in a], [event.t_s for event in b])
        self.assertEqual([event.xyz_mm for event in a], [event.xyz_mm for event in b])

    def test_coordinates_stay_in_region(self) -> None:
        generator = SyntheticGenerator(seed=7)
        events = generator.generate(0.0, 0.5)
        self.assertTrue(events)
        for event in events:
            region = get_region(event.region_id)
            self.assertTrue(region.contains(event.xyz_mm, margin=1.08), event.region_id)

    def test_not_uniform_whole_brain(self) -> None:
        generator = SyntheticGenerator(seed=3)
        counts: dict[str, int] = {}
        for event in generator.generate(0.0, 1.0):
            counts[event.region_id] = counts.get(event.region_id, 0) + 1
        self.assertGreaterEqual(len(counts), 4)
        self.assertIn("visual_cortex", counts)

    def test_pause_freezes_clock(self) -> None:
        engine = NeuralEventEngine()
        engine.control({"reset": True, "seed": 11, "run_golden_model": False})
        engine.control({"running": True})
        engine.step_wall(0.05)
        t = engine.snapshot()["t_s"]
        self.assertGreater(t, 0.0)
        engine.control({"running": False})
        engine.step_wall(0.05)
        self.assertAlmostEqual(engine.snapshot()["t_s"], t)


if __name__ == "__main__":
    unittest.main()
