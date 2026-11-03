"""Software FIFO for the emulated ASIC-to-edge link.

Handshake is ready/valid style:

- ``ready`` is high when ``try_push`` can accept a frame without dropping.
- ``valid`` is high when ``pop`` / ``try_pop`` can return a frame.
- ``try_push`` returns True on accept (valid beat) and False on backpressure.

When ``drop_oldest`` is True (the schema default), a push into a full FIFO
discards the oldest queued frame, counts it as overflow loss, and accepts the
new frame. That is *not* backpressure; the drop is silent on the wire and the
receiver infers loss from sequence gaps. When ``drop_oldest`` is False, a full
FIFO deasserts ready and ``try_push`` returns False; occupancy is unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from edge.link.codec import PacketError, header_seq
from edge.link.schema import DEFAULT_FIFO_DEPTH, DROP_OLDEST, HEADER_SIZE, HEADER_STRUCT


@dataclass(frozen=True)
class GapEvent:
    """Observed sequence skip: packets ``from_seq`` through ``to_seq`` inclusive were missing."""

    expected_seq: int
    observed_seq: int
    missing: int

    @property
    def from_seq(self) -> int:
        return int(self.expected_seq)

    @property
    def to_seq(self) -> int:
        return int(self.observed_seq - 1) & 0xFFFFFFFF


class SequenceTracker:
    """Detects lost packets from the 32-bit header sequence number."""

    def __init__(self) -> None:
        self.last_seq: int | None = None
        self.observed = 0
        self.gap_count = 0
        self.missing_packets = 0
        self.gaps: list[GapEvent] = []

    def reset(self) -> None:
        self.last_seq = None
        self.observed = 0
        self.gap_count = 0
        self.missing_packets = 0
        self.gaps.clear()

    def observe(self, seq: int) -> GapEvent | None:
        seq = int(seq) & 0xFFFFFFFF
        self.observed += 1
        if self.last_seq is None:
            self.last_seq = seq
            return None
        expected = (self.last_seq + 1) & 0xFFFFFFFF
        self.last_seq = seq
        if seq == expected:
            return None
        missing = (seq - expected) & 0xFFFFFFFF
        # A full 2**32 skip is treated as one wrap of missing packets.
        event = GapEvent(expected_seq=expected, observed_seq=seq, missing=missing)
        self.gap_count += 1
        self.missing_packets += missing
        self.gaps.append(event)
        return event


class LinkFifo:
    """Fixed-depth byte-frame FIFO with optional drop-oldest overflow."""

    def __init__(self, depth: int = DEFAULT_FIFO_DEPTH, drop_oldest: bool = DROP_OLDEST) -> None:
        if depth <= 0:
            raise ValueError("FIFO depth must be positive")
        self.depth = int(depth)
        self.drop_oldest = bool(drop_oldest)
        self._queue: deque[bytes] = deque()
        self.pushed = 0
        self.popped = 0
        self.dropped = 0
        self.overflow_count = 0
        self.backpressure_count = 0
        self.tracker = SequenceTracker()
        self.track_on_push = True

    def reset(self) -> None:
        self._queue.clear()
        self.pushed = 0
        self.popped = 0
        self.dropped = 0
        self.overflow_count = 0
        self.backpressure_count = 0
        self.tracker.reset()

    @property
    def occupancy(self) -> int:
        return len(self._queue)

    @property
    def free(self) -> int:
        return self.depth - len(self._queue)

    @property
    def ready(self) -> bool:
        """Input ready: a push will not be rejected. True when not full, or drop-oldest is enabled."""
        return self.drop_oldest or len(self._queue) < self.depth

    @property
    def valid(self) -> bool:
        """Output valid: a pop will return a frame."""
        return bool(self._queue)

    @property
    def full(self) -> bool:
        return len(self._queue) >= self.depth

    @property
    def empty(self) -> bool:
        return not self._queue

    def try_push(self, frame: bytes) -> bool:
        """Attempt a valid/ready push. Returns False under backpressure (full, no drop)."""
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise TypeError("FIFO stores framed bytes")
        blob = bytes(frame)
        if len(blob) < HEADER_SIZE:
            raise PacketError("truncated frame")
        if len(self._queue) >= self.depth:
            if not self.drop_oldest:
                self.backpressure_count += 1
                return False
            self._queue.popleft()
            self.dropped += 1
            self.overflow_count += 1
        self._queue.append(blob)
        self.pushed += 1
        if self.track_on_push:
            self.tracker.observe(header_seq(blob))
        return True

    def try_pop(self) -> bytes | None:
        """Pop the oldest frame, or None when not valid."""
        if not self._queue:
            return None
        frame = self._queue.popleft()
        self.popped += 1
        if not self.track_on_push:
            self.tracker.observe(header_seq(frame))
        return frame

    def pop(self) -> bytes:
        frame = self.try_pop()
        if frame is None:
            raise PacketError("FIFO empty")
        return frame

    def peek(self) -> bytes | None:
        if not self._queue:
            return None
        return self._queue[0]

    def status_args(self) -> tuple[int, ...]:
        """Integer snapshot used by GET_STATUS ACKs."""
        header_flags = 0
        if self._queue:
            _m, _v, _t, header_flags, _s, _p = HEADER_STRUCT.unpack(self._queue[-1][:HEADER_SIZE])
        return (
            self.occupancy,
            self.depth,
            self.dropped,
            self.overflow_count,
            self.backpressure_count,
            self.tracker.gap_count,
            self.tracker.missing_packets,
            1 if self.full else 0,
            int(header_flags) & 0xFF,
        )
