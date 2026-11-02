from __future__ import annotations

import json
from pathlib import Path
import unittest

from viz.sim.events import REQUIRED_FIELDS, SCHEMA_VERSION, NeuralEvent

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema" / "neural_event.v1.json").read_text())


class SchemaTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        event = NeuralEvent(
            event_id="syn:demo:0",
            t_s=0.12,
            xyz_mm=(1.0, -2.0, 3.5),
            region_id="visual_cortex",
            amplitude=-88.0,
            source_id="demo",
            kind="synthetic_source",
        )
        restored = NeuralEvent.from_dict(event.to_dict())
        self.assertEqual(restored, event)
        self.assertEqual(restored.schema_version, SCHEMA_VERSION)

    def test_required_fields_match_schema(self) -> None:
        self.assertEqual(set(REQUIRED_FIELDS), set(SCHEMA["required"]))
        self.assertEqual(SCHEMA["properties"]["schema_version"]["const"], 1)

    def test_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            NeuralEvent(
                event_id="x",
                t_s=0.0,
                xyz_mm=(0.0, 0.0, 0.0),
                region_id="thalamus",
                amplitude=1.0,
                source_id="x",
                kind="made_up",
            )


if __name__ == "__main__":
    unittest.main()
