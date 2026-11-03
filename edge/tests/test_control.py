from __future__ import annotations

import unittest

from edge.link.buffer import LinkFifo
from edge.link.codec import CmdPacket, decode_packet, encode_ack, encode_cmd
from edge.link.control import STATUS_BAD_MODE, STATUS_OK, ControlPlane
from edge.link.schema import (
    ACQ_BOTH,
    ACQ_EVENT,
    ACQ_SNIPPET,
    CMD_DIAG,
    CMD_GET_STATUS,
    CMD_PING,
    CMD_SET_CHANNELS,
    CMD_SET_MODE,
    CMD_SET_THRESHOLD,
)


class ControlPlaneTests(unittest.TestCase):
    def test_set_threshold_broadcast_and_per_channel(self) -> None:
        plane = ControlPlane(n_channels=4)
        ack = plane.apply(CmdPacket(opcode=CMD_SET_THRESHOLD, args=(-1200,), seq=1))
        self.assertEqual(ack.status, STATUS_OK)
        self.assertEqual(plane.thresholds, [-1200, -1200, -1200, -1200])
        ack = plane.apply(CmdPacket(opcode=CMD_SET_THRESHOLD, args=(2, -50), seq=2))
        self.assertEqual(ack.seq, 2)
        self.assertEqual(plane.thresholds[2], -50)
        ack = plane.apply(CmdPacket(opcode=CMD_SET_THRESHOLD, args=(-1, -2, -3, -4), seq=3))
        self.assertEqual(plane.thresholds, [-1, -2, -3, -4])
        self.assertEqual(decode_packet(encode_ack(ack)).args, (-1, -2, -3, -4))

    def test_set_channels_mask_and_bit_list(self) -> None:
        plane = ControlPlane(n_channels=8)
        ack = plane.apply(CmdPacket(opcode=CMD_SET_CHANNELS, args=(0b0000_0101,)))
        self.assertEqual(ack.status, STATUS_OK)
        self.assertEqual(plane.enabled, [True, False, True, False, False, False, False, False])
        ack = plane.apply(CmdPacket(opcode=CMD_SET_CHANNELS, args=(0, 1, 1, 0, 0, 0, 0, 1)))
        self.assertEqual(plane.enabled[-1], True)
        self.assertEqual(plane.enabled[0], False)
        self.assertEqual(sum(plane.enabled), 3)

    def test_set_mode_status_and_ping(self) -> None:
        fifo = LinkFifo(depth=4, drop_oldest=False)
        plane = ControlPlane(n_channels=4, fifo=fifo)
        ack = plane.apply(CmdPacket(opcode=CMD_SET_MODE, args=(ACQ_EVENT,), seq=4))
        self.assertEqual(ack.status, STATUS_OK)
        self.assertEqual(plane.mode, ACQ_EVENT)
        bad = plane.apply(CmdPacket(opcode=CMD_SET_MODE, args=(99,), seq=5))
        self.assertEqual(bad.status, STATUS_BAD_MODE)
        ping = plane.apply(CmdPacket(opcode=CMD_PING, args=(7, 8), seq=6))
        self.assertEqual(ping.status, STATUS_OK)
        self.assertEqual(ping.args, (7, 8))
        self.assertEqual(decode_packet(encode_cmd(CmdPacket(opcode=CMD_PING, args=(7, 8), seq=6))).args, (7, 8))
        status = plane.apply(CmdPacket(opcode=CMD_GET_STATUS, seq=7))
        self.assertEqual(status.status, STATUS_OK)
        self.assertEqual(status.args[0], ACQ_EVENT)
        self.assertEqual(status.args[1], 4)
        diag = plane.apply(CmdPacket(opcode=CMD_DIAG, seq=8))
        self.assertEqual(diag.status, STATUS_OK)
        self.assertEqual(diag.args[1], 4)
        self.assertEqual(diag.args[3], 0)  # no stimulation payload
        plane.apply(CmdPacket(opcode=CMD_SET_MODE, args=(ACQ_SNIPPET,)))
        self.assertEqual(plane.mode, ACQ_SNIPPET)
        plane.apply(CmdPacket(opcode=CMD_SET_MODE, args=(ACQ_BOTH,)))
        self.assertEqual(plane.mode, ACQ_BOTH)


if __name__ == "__main__":
    unittest.main()
