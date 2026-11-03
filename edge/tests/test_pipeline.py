from __future__ import annotations

import unittest

import numpy as np

from edge.decode.features import SPATIAL_DIM
from edge.decode.model import train
from edge.decode.spatial_task import generate_dataset, jitter_estimates_from_ground_truth, snippets_for_trial
from edge.pipeline import DecodePrediction, EventAssembler, StreamEvent, ingest, run_pipeline
from edge.types import LocalizationEstimate


class PipelineTests(unittest.TestCase):
    def test_event_and_snippet_pair_in_either_order(self) -> None:
        from edge.link.codec import EventPacket, SnippetPacket, encode_packet
        from edge.link.schema import FLAG_HAS_SNIPPET
        event = EventPacket(19, 3200, 3, -20, flags=FLAG_HAS_SNIPPET, seq=40)
        snippet = SnippetPacket(19, 3200, 3198, 32000.0, (2, 3),
                                np.array([[1, 2], [3, 4], [-5, -20]], dtype=np.int16),
                                3, 0.1, seq=41)
        for packets in ((event, snippet), (snippet, event)):
            assembler = EventAssembler()
            for packet in packets:
                assembler.feed(encode_packet(packet))
            found = assembler.finish()
            self.assertEqual(len(found), 1)
            self.assertEqual((found[0].event_id, found[0].t_sample, found[0].channel_id), (19, 3200, 3))
            self.assertEqual(found[0].snippet.channel_ids, (2, 3))
            np.testing.assert_allclose(found[0].snippet.voltages_uv,
                                       snippet.samples.astype(float) * float(np.float32(0.1)))
            self.assertEqual(assembler.diagnostics.missing_packets, 0)

    def test_missing_duplicate_conflict_and_corrupt_packets(self) -> None:
        from edge.link.codec import EventPacket, PacketError, encode_packet
        from edge.link.schema import FLAG_HAS_SNIPPET
        event = EventPacket(1, 10, 2, -4, flags=FLAG_HAS_SNIPPET, seq=3)
        assembler = EventAssembler(corrupt="skip")
        assembler.feed(encode_packet(event))
        assembler.feed(encode_packet(EventPacket(1, 10, 2, -4, flags=FLAG_HAS_SNIPPET, seq=5)))
        bad = bytearray(encode_packet(EventPacket(2, 20, 1, -2, seq=6)))
        bad[-1] ^= 1
        assembler.feed(bytes(bad))
        self.assertEqual(len(assembler.finish()), 1)
        self.assertEqual(assembler.diagnostics.duplicate_packets, 1)
        self.assertEqual(assembler.diagnostics.missing_packets, 1)
        self.assertEqual(assembler.diagnostics.missing_snippets, 1)
        self.assertEqual(assembler.diagnostics.corrupt_packets, 1)
        with self.assertRaises(PacketError):
            ingest([bytes(bad)])
        with self.assertRaises(ValueError):
            ingest([event, EventPacket(1, 10, 2, -5, seq=4)])
        with self.assertRaises(ValueError):
            ingest([event, EventPacket(2, 20, 1, -2, seq=3)])

    def test_bounded_session_and_sequence_wrap(self) -> None:
        from edge.link.codec import EventPacket
        assembler = EventAssembler(max_events=1)
        assembler.feed(EventPacket(1, 10, 2, -4, seq=0))
        with self.assertRaises(OverflowError):
            assembler.feed(EventPacket(2, 20, 3, -5, seq=1))
        wrap = EventAssembler()
        for seq in (0, 0xFFFFFFFF, 1):
            wrap.feed(EventPacket(seq, seq, 0, -1, seq=seq))
        self.assertEqual(wrap.diagnostics.missing_packets, 0)
        self.assertEqual(len(wrap.finish()), 3)
        self.assertEqual(wrap.diagnostics.missing_packets, 0)

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
