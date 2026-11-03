from __future__ import annotations

import unittest

from edge.link.buffer import LinkFifo, SequenceTracker
from edge.link.codec import EventPacket, encode_event
from edge.link.schema import DEFAULT_FIFO_DEPTH, DROP_OLDEST


def _event_frame(seq: int) -> bytes:
    return encode_event(EventPacket(event_id=seq, t_sample=seq, channel_id=0, amplitude=0, seq=seq))


class LinkBufferTests(unittest.TestCase):
    def test_ready_valid_and_backpressure(self) -> None:
        fifo = LinkFifo(depth=2, drop_oldest=False)
        self.assertTrue(fifo.ready)
        self.assertFalse(fifo.valid)
        self.assertTrue(fifo.try_push(_event_frame(0)))
        self.assertTrue(fifo.try_push(_event_frame(1)))
        self.assertTrue(fifo.full)
        self.assertFalse(fifo.ready)
        self.assertFalse(fifo.try_push(_event_frame(2)))
        self.assertEqual(fifo.occupancy, 2)
        self.assertEqual(fifo.backpressure_count, 1)
        self.assertEqual(fifo.dropped, 0)
        self.assertEqual(fifo.overflow_count, 0)
        from edge.link.codec import decode_packet

        popped = fifo.try_pop()
        self.assertIsNotNone(popped)
        self.assertEqual(decode_packet(popped).seq, 0)
        self.assertEqual(fifo.occupancy, 1)
        self.assertTrue(fifo.ready)
        self.assertTrue(fifo.try_push(_event_frame(2)))

    def test_drop_oldest_overflow_counts(self) -> None:
        self.assertTrue(DROP_OLDEST)
        fifo = LinkFifo(depth=DEFAULT_FIFO_DEPTH, drop_oldest=True)
        for seq in range(DEFAULT_FIFO_DEPTH):
            self.assertTrue(fifo.try_push(_event_frame(seq)))
        self.assertTrue(fifo.full)
        self.assertTrue(fifo.ready)  # drop-oldest keeps ready asserted
        self.assertTrue(fifo.try_push(_event_frame(DEFAULT_FIFO_DEPTH)))
        self.assertEqual(fifo.occupancy, DEFAULT_FIFO_DEPTH)
        self.assertEqual(fifo.dropped, 1)
        self.assertEqual(fifo.overflow_count, 1)
        self.assertEqual(fifo.backpressure_count, 0)
        first = fifo.pop()
        from edge.link.codec import decode_packet

        restored = decode_packet(first)
        self.assertEqual(restored.seq, 1)

    def test_sequence_gap_detected(self) -> None:
        tracker = SequenceTracker()
        self.assertIsNone(tracker.observe(0))
        self.assertIsNone(tracker.observe(1))
        gap = tracker.observe(4)
        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertEqual(gap.expected_seq, 2)
        self.assertEqual(gap.observed_seq, 4)
        self.assertEqual(gap.missing, 2)
        self.assertEqual(tracker.gap_count, 1)
        self.assertEqual(tracker.missing_packets, 2)

        fifo = LinkFifo(depth=8, drop_oldest=True)
        self.assertTrue(fifo.try_push(_event_frame(10)))
        self.assertTrue(fifo.try_push(_event_frame(11)))
        self.assertTrue(fifo.try_push(_event_frame(13)))
        self.assertEqual(fifo.tracker.gap_count, 1)
        self.assertEqual(fifo.tracker.missing_packets, 1)
        self.assertEqual(fifo.tracker.gaps[0].expected_seq, 12)


if __name__ == "__main__":
    unittest.main()
