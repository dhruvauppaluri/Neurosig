from __future__ import annotations

import unittest

import numpy as np

from edge.decode.features import SPATIAL_DIM
from edge.decode.model import train
from edge.decode.spatial_task import generate_dataset, jitter_estimates_from_ground_truth, snippets_for_trial
from edge.pipeline import DecodePrediction, StreamEvent, ingest, run_pipeline
from edge.types import LocalizationEstimate


class PipelineTests(unittest.TestCase):
    def test_ingest_snippets_and_returns_predictions(self) -> None:
        dataset = generate_dataset(seed=20261103, examples=8)
        snippets = snippets_for_trial(dataset, trial=0)
        self.assertGreater(len(snippets), 0)
        weights = np.zeros(32, dtype=np.float64)
        bias = 0.0
        predictions = run_pipeline(snippets, weights, bias)
        self.assertGreater(len(predictions), 0)
        self.assertIsInstance(predictions[0], DecodePrediction)
        self.assertEqual(predictions[0].label in (0, 1), True)
        self.assertGreaterEqual(predictions[0].confidence, 0.0)
        self.assertLessEqual(predictions[0].confidence, 1.0)
        self.assertGreater(predictions[0].n_events, 0)
        self.assertIsInstance(predictions[0].used_localization, bool)
        from edge.localize.base import NotImplementedLocalizer
        skipped = run_pipeline(snippets, weights, bias, localizer=NotImplementedLocalizer())
        self.assertFalse(skipped[0].used_localization)

    def test_pipeline_with_jitter_estimates(self) -> None:
        dataset = generate_dataset(seed=20261103, examples=16)
        snippets = snippets_for_trial(dataset, trial=1)
        rng = np.random.default_rng(20261103)
        estimates = jitter_estimates_from_ground_truth(snippets, rng)
        items: list[object] = []
        for snippet, estimate in zip(snippets, estimates):
            items.append(snippet)
            items.append(estimate)
        events = ingest(items)
        self.assertEqual(len(events), len(snippets))
        self.assertTrue(any(event.localization is not None for event in events))
        spatial_x = np.zeros((8, SPATIAL_DIM), dtype=np.float64)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        spatial_x[:, 0] = labels * 5
        weights, bias = train(spatial_x, labels)
        predictions = run_pipeline(items, weights, bias)
        self.assertGreater(len(predictions), 0)
        self.assertTrue(all(isinstance(p.score, float) for p in predictions))
        self.assertTrue(predictions[0].used_localization)

    def test_bytes_packets_use_codec_or_typed_objects(self) -> None:
        try:
            from edge.link.codec import EventPacket, PacketError, encode_event
        except ImportError:
            with self.assertRaises(TypeError):
                ingest([b"NSG1notapacket"])
            return
        with self.assertRaises(PacketError):
            ingest([b"NSG1notapacket"])
        frame = encode_event(EventPacket(event_id=7, t_sample=3200, channel_id=3, amplitude=-12))
        events = ingest([frame])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, 7)
        self.assertEqual(events[0].channel_id, 3)

    def test_stream_event_passthrough(self) -> None:
        event = StreamEvent(event_id=1, t_sample=100, channel_id=4)
        predictions = run_pipeline([event], np.zeros(32), 1.0)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].label, 1)
        self.assertAlmostEqual(predictions[0].confidence, 1.0 / (1.0 + np.exp(-1.0)))

    def test_localization_estimate_without_event_id_ignored(self) -> None:
        estimate = LocalizationEstimate(
            available=True,
            method="test",
            xyz_um=(0.0, 100.0, 0.0),
            ground_truth_xyz_um=None,
            error_um=None,
            latency_s=0.0,
            confidence=None,
            reason="unit test",
            event_id=None,
        )
        self.assertEqual(ingest([estimate]), [])


if __name__ == "__main__":
    unittest.main()
