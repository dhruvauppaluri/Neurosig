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
        localization = body["localization"]
        for key in ("available", "method", "error_um", "confidence", "unverified", "reason", "xyz_mm"):
            self.assertIn(key, localization)
        self.assertIn(localization["available"], (True, False))

    def test_control_pause_stops_timestamp(self) -> None:
        server.engine.control({"running": True})
        server.engine.step_wall(0.08)
        paused = self.client.post("/api/v1/session/control", json={"running": False}).json()
        t = paused["t_s"]
        server.engine.step_wall(0.2)
        self.assertAlmostEqual(server.engine.snapshot()["t_s"], t)

    def test_snippet_endpoint(self) -> None:
        missing = self.client.get("/api/v1/snippet/does-not-exist")
        self.assertEqual(missing.status_code, 404)
        server.engine.control({"reset": True, "seed": 5, "run_golden_model": True, "mode": "synthetic", "running": True})
        for _ in range(10):
            server.engine.step_wall(0.08)
        if not server.engine.snippets:
            return
        event_id = next(iter(server.engine.snippets))
        body = self.client.get(f"/api/v1/snippet/{event_id}").json()
        for key in ("event_id", "sample_rate_hz", "channel_ids", "t0_sample", "t_sample", "voltages_uv"):
            self.assertIn(key, body)
        self.assertIsInstance(body["voltages_uv"], list)

    def test_mode_switch(self) -> None:
        body = self.client.post("/api/v1/session/control", json={"mode": "neuropixels"}).json()
        self.assertEqual(body["mode"], "neuropixels")
        self.assertIn("Mus musculus", body["neuropixels"]["species"])


if __name__ == "__main__":
    unittest.main()
