from __future__ import annotations

import struct
import unittest

import numpy as np

from edge.link.codec import (
    AckPacket,
    CmdPacket,
    EventPacket,
    PacketError,
    SnippetPacket,
    crc32,
    decode_packet,
    encode_ack,
    encode_cmd,
    encode_event,
    encode_snippet,
)
from edge.link.schema import (
    CMD_PING,
    CRC_SIZE,
    EVENT_PAYLOAD_SIZE,
    FLAG_HAS_SNIPPET,
    HEADER_SIZE,
    MAGIC,
    PKT_EVENT,
    SCHEMA_VERSION,
)


class LinkCodecTests(unittest.TestCase):
    def test_event_round_trip(self) -> None:
        packet = EventPacket(
            event_id=7,
            t_sample=32000,
            channel_id=21,
            amplitude=-1234,
            flags=FLAG_HAS_SNIPPET,
            reserved=0,
            seq=42,
            header_flags=0,
        )
        frame = encode_event(packet)
        restored = decode_packet(frame)
        self.assertIsInstance(restored, EventPacket)
        self.assertEqual(restored.event_id, 7)
        self.assertEqual(restored.t_sample, 32000)
        self.assertEqual(restored.channel_id, 21)
        self.assertEqual(restored.amplitude, -1234)
        self.assertEqual(restored.flags, FLAG_HAS_SNIPPET)
        self.assertEqual(restored.seq, 42)
        self.assertEqual(frame[:4], MAGIC)
        version = struct.unpack_from("<H", frame, 4)[0]
        self.assertEqual(version, SCHEMA_VERSION)
        ptype = frame[6]
        self.assertEqual(ptype, PKT_EVENT)
        payload_len = struct.unpack_from("<I", frame, 12)[0]
        self.assertEqual(payload_len, EVENT_PAYLOAD_SIZE)
        self.assertEqual(len(frame), HEADER_SIZE + payload_len + CRC_SIZE)

    def test_snippet_round_trip_sample_major(self) -> None:
        samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int16)
        packet = SnippetPacket(
            event_id=3,
            t_sample=80,
            t0_sample=48,
            sample_rate_hz=32_000.0,
            channel_ids=(10, 11, 12),
            samples=samples,
            peak_channel=11,
            lsb_uv=0.004119873046875,
            seq=9,
        )
        restored = decode_packet(encode_snippet(packet))
        self.assertIsInstance(restored, SnippetPacket)
        self.assertEqual(restored.event_id, 3)
        self.assertEqual(restored.t_sample, 80)
        self.assertEqual(restored.t0_sample, 48)
        self.assertEqual(restored.peak_index, 32)
        self.assertEqual(restored.channel_ids, (10, 11, 12))
        self.assertEqual(restored.peak_channel, 11)
        np.testing.assert_array_equal(restored.samples, samples)
        self.assertAlmostEqual(restored.sample_rate_hz, 32_000.0)
        self.assertAlmostEqual(restored.lsb_uv, 0.004119873046875)
        # Wire order is sample-major: ch0,ch1,ch2 of sample 0, then sample 1, ...
        payload = encode_snippet(packet)[HEADER_SIZE:-CRC_SIZE]
        raw = payload[34 + 6 :]
        unpacked = np.frombuffer(raw, dtype="<i2")
        np.testing.assert_array_equal(unpacked, samples.reshape(-1))

    def test_crc_fail_rejected(self) -> None:
        packet = EventPacket(event_id=1, t_sample=2, channel_id=3, amplitude=4, seq=5)
        frame = bytearray(encode_event(packet))
        frame[HEADER_SIZE] ^= 0xFF
        with self.assertRaises(PacketError) as ctx:
            decode_packet(bytes(frame))
        self.assertIn("CRC32", str(ctx.exception))

    def test_bad_magic_rejected(self) -> None:
        frame = bytearray(encode_event(EventPacket(event_id=0, t_sample=0, channel_id=0, amplitude=0)))
        frame[0:4] = b"XXXX"
        crc = crc32(bytes(frame[: len(frame) - CRC_SIZE]))
        frame[-4:] = struct.pack("<I", crc)
        with self.assertRaises(PacketError):
            decode_packet(bytes(frame))

    def test_cmd_ack_round_trip(self) -> None:
        cmd = CmdPacket(opcode=CMD_PING, args=(-1, 2, 3), seq=11)
        ack = AckPacket(opcode=CMD_PING, status=0, args=(-1, 2, 3), seq=11)
        self.assertEqual(decode_packet(encode_cmd(cmd)), cmd)
        self.assertEqual(decode_packet(encode_ack(ack)), ack)


if __name__ == "__main__":
    unittest.main()
