from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from viz.sim import server


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        server.engine.control({"reset": True, "seed": 5, "run_golden_model": False, "mode": "synthetic"})
        self.client = TestClient(server.app)

    def test_session_endpoint(self) -> None:
        response = self.client.get("/api/v1/session")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertIn("disclaimer", body)
        self.assertFalse(body["localization"]["available"])

    def test_control_pause_stops_timestamp(self) -> None:
        server.engine.control({"running": True})
        server.engine.step_wall(0.08)
        paused = self.client.post("/api/v1/session/control", json={"running": False}).json()
        t = paused["t_s"]
        server.engine.step_wall(0.2)
        self.assertAlmostEqual(server.engine.snapshot()["t_s"], t)

    def test_mode_switch(self) -> None:
        body = self.client.post("/api/v1/session/control", json={"mode": "neuropixels"}).json()
        self.assertEqual(body["mode"], "neuropixels")
        self.assertIn("Mus musculus", body["neuropixels"]["species"])


if __name__ == "__main__":
    unittest.main()
