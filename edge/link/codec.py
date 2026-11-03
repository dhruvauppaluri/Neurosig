"""Encode and decode versioned NeuroSig ASIC-to-edge packets.

Wire image is ``header || payload || crc32_le``. CRC32 is IEEE 802.3
(``binascii.crc32``) over every byte before the CRC field. This module is a
software codec only; it does not implement RTL.
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass

import numpy as np

from edge.link.schema import (
    CMD_PREFIX_SIZE,
    CMD_PREFIX_STRUCT,
    CRC_SIZE,
    EVENT_PAYLOAD_SIZE,
    EVENT_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    PKT_ACK,
    PKT_CMD,
    PKT_EVENT,
    PKT_SNIPPET,
    SCHEMA_VERSION,
    SNIPPET_PREFIX_SIZE,
    SNIPPET_PREFIX_STRUCT,
)


class PacketError(ValueError):
    """Raised when a frame is truncated, mistyped, or fails integrity checks."""


@dataclass(frozen=True)
class EventPacket:
    event_id: int
    t_sample: int
    channel_id: int
    amplitude: int
    flags: int = 0
    reserved: int = 0
    seq: int = 0
    header_flags: int = 0


@dataclass(frozen=True)
class SnippetPacket:
    event_id: int
    t_sample: int
    t0_sample: int
    sample_rate_hz: float
    channel_ids: tuple[int, ...]
    samples: np.ndarray
    peak_channel: int
    lsb_uv: float
    seq: int = 0
    header_flags: int = 0

    @property
    def n_ch(self) -> int:
        return len(self.channel_ids)

    @property
    def n_samp(self) -> int:
        return int(self.samples.shape[0]) if self.samples.ndim == 2 else 0

    @property
    def peak_index(self) -> int:
        """Sample-major row of the event peak: ``t_sample - t0_sample``."""
        return int(self.t_sample - self.t0_sample)


@dataclass(frozen=True)
class CmdPacket:
    opcode: int
    args: tuple[int, ...] = ()
    status: int = 0
    reserved: int = 0
    seq: int = 0
    header_flags: int = 0


@dataclass(frozen=True)
class AckPacket:
    opcode: int
    status: int = 0
    args: tuple[int, ...] = ()
    reserved: int = 0
    seq: int = 0
    header_flags: int = 0


Packet = EventPacket | SnippetPacket | CmdPacket | AckPacket


def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def header_seq(frame: bytes) -> int:
    """Read ``seq`` from a framed packet without validating CRC."""
    if len(frame) < HEADER_SIZE:
        raise PacketError("truncated header")
    _magic, _version, _ptype, _flags, seq, _payload_len = HEADER_STRUCT.unpack(frame[:HEADER_SIZE])
    return int(seq)


def encode_event(packet: EventPacket) -> bytes:
    payload = EVENT_STRUCT.pack(
        int(packet.event_id) & 0xFFFFFFFF,
        int(packet.t_sample) & 0xFFFFFFFFFFFFFFFF,
        int(packet.channel_id) & 0xFFFF,
        int(packet.amplitude),
        int(packet.flags) & 0xFFFF,
        int(packet.reserved) & 0xFFFF,
    )
    return _frame(PKT_EVENT, payload, packet.seq, packet.header_flags)


def encode_snippet(packet: SnippetPacket) -> bytes:
    samples = np.asarray(packet.samples, dtype=np.int16)
    if samples.ndim != 2:
        raise PacketError("snippet samples must be [n_samp, n_ch]")
    n_samp, n_ch = samples.shape
    if n_ch != len(packet.channel_ids):
        raise PacketError("channel_ids length does not match sample columns")
    prefix = SNIPPET_PREFIX_STRUCT.pack(
        int(packet.event_id) & 0xFFFFFFFF,
        int(packet.t_sample) & 0xFFFFFFFFFFFFFFFF,
        int(packet.t0_sample) & 0xFFFFFFFFFFFFFFFF,
        float(packet.sample_rate_hz),
        n_ch & 0xFFFF,
        n_samp & 0xFFFF,
        int(packet.peak_channel),
        float(packet.lsb_uv),
    )
    channel_bytes = struct.pack(f"<{n_ch}H", *[int(c) & 0xFFFF for c in packet.channel_ids])
    # Sample-major: all channels of sample 0, then sample 1, ...
    sample_bytes = samples.astype("<i2", copy=False).reshape(-1).tobytes()
    return _frame(PKT_SNIPPET, prefix + channel_bytes + sample_bytes, packet.seq, packet.header_flags)


def encode_cmd(packet: CmdPacket) -> bytes:
    return _frame(PKT_CMD, _pack_cmd(packet.opcode, packet.status, packet.args, packet.reserved), packet.seq, packet.header_flags)


def encode_ack(packet: AckPacket) -> bytes:
    return _frame(PKT_ACK, _pack_cmd(packet.opcode, packet.status, packet.args, packet.reserved), packet.seq, packet.header_flags)


def encode_packet(packet: Packet) -> bytes:
    if isinstance(packet, EventPacket):
        return encode_event(packet)
    if isinstance(packet, SnippetPacket):
        return encode_snippet(packet)
    if isinstance(packet, CmdPacket):
        return encode_cmd(packet)
    if isinstance(packet, AckPacket):
        return encode_ack(packet)
    raise PacketError(f"unsupported packet type {type(packet)!r}")


def decode_packet(frame: bytes) -> Packet:
    """Decode one framed packet. Raises ``PacketError`` on any corruption."""
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise PacketError("frame must be bytes")
    blob = bytes(frame)
    if len(blob) < HEADER_SIZE + CRC_SIZE:
        raise PacketError("truncated frame")

    magic, version, ptype, header_flags, seq, payload_len = HEADER_STRUCT.unpack(blob[:HEADER_SIZE])
    if magic != MAGIC:
        raise PacketError("invalid magic")
    if version != SCHEMA_VERSION:
        raise PacketError("unsupported schema version")
    if payload_len > len(blob):
        raise PacketError("payload_len exceeds frame")
    expected = HEADER_SIZE + payload_len + CRC_SIZE
    if len(blob) != expected:
        raise PacketError("frame length does not match payload_len")

    payload = blob[HEADER_SIZE : HEADER_SIZE + payload_len]
    body = blob[: HEADER_SIZE + payload_len]
    crc_got = struct.unpack("<I", blob[HEADER_SIZE + payload_len :])[0]
    if crc_got != crc32(body):
        raise PacketError("CRC32 mismatch")

    if ptype == PKT_EVENT:
        return _decode_event(payload, seq, header_flags)
    if ptype == PKT_SNIPPET:
        return _decode_snippet(payload, seq, header_flags)
    if ptype == PKT_CMD:
        opcode, status, args, reserved = _unpack_cmd(payload)
        return CmdPacket(opcode=opcode, args=args, status=status, reserved=reserved, seq=seq, header_flags=header_flags)
    if ptype == PKT_ACK:
        opcode, status, args, reserved = _unpack_cmd(payload)
        return AckPacket(opcode=opcode, status=status, args=args, reserved=reserved, seq=seq, header_flags=header_flags)
    raise PacketError(f"unknown packet type {ptype}")


def _frame(ptype: int, payload: bytes, seq: int, header_flags: int) -> bytes:
    header = HEADER_STRUCT.pack(
        MAGIC,
        SCHEMA_VERSION,
        int(ptype) & 0xFF,
        int(header_flags) & 0xFF,
        int(seq) & 0xFFFFFFFF,
        len(payload),
    )
    body = header + payload
    return body + struct.pack("<I", crc32(body))


def _pack_cmd(opcode: int, status: int, args: tuple[int, ...] | list[int], reserved: int) -> bytes:
    packed_args = tuple(int(a) for a in args)
    n_args = len(packed_args)
    prefix = CMD_PREFIX_STRUCT.pack(int(opcode) & 0xFFFF, int(status) & 0xFFFF, n_args & 0xFFFF, int(reserved) & 0xFFFF)
    if n_args == 0:
        return prefix
    return prefix + struct.pack(f"<{n_args}i", *packed_args)


def _unpack_cmd(payload: bytes) -> tuple[int, int, tuple[int, ...], int]:
    if len(payload) < CMD_PREFIX_SIZE:
        raise PacketError("CMD/ACK payload shorter than prefix")
    opcode, status, n_args, reserved = CMD_PREFIX_STRUCT.unpack(payload[:CMD_PREFIX_SIZE])
    expected = CMD_PREFIX_SIZE + 4 * n_args
    if len(payload) != expected:
        raise PacketError("CMD/ACK payload_len does not match n_args")
    args: tuple[int, ...]
    if n_args:
        args = struct.unpack(f"<{n_args}i", payload[CMD_PREFIX_SIZE:])
    else:
        args = ()
    return int(opcode), int(status), args, int(reserved)


def _decode_event(payload: bytes, seq: int, header_flags: int) -> EventPacket:
    if len(payload) != EVENT_PAYLOAD_SIZE:
        raise PacketError("EVENT payload_len mismatch")
    event_id, t_sample, channel_id, amplitude, flags, reserved = EVENT_STRUCT.unpack(payload)
    return EventPacket(
        event_id=int(event_id),
        t_sample=int(t_sample),
        channel_id=int(channel_id),
        amplitude=int(amplitude),
        flags=int(flags),
        reserved=int(reserved),
        seq=int(seq),
        header_flags=int(header_flags),
    )


def _decode_snippet(payload: bytes, seq: int, header_flags: int) -> SnippetPacket:
    if len(payload) < SNIPPET_PREFIX_SIZE:
        raise PacketError("SNIPPET payload shorter than prefix")
    (
        event_id,
        t_sample,
        t0_sample,
        sample_rate_hz,
        n_ch,
        n_samp,
        peak_channel,
        lsb_uv,
    ) = SNIPPET_PREFIX_STRUCT.unpack(payload[:SNIPPET_PREFIX_SIZE])
    channel_bytes = 2 * n_ch
    sample_bytes = 2 * n_ch * n_samp
    expected = SNIPPET_PREFIX_SIZE + channel_bytes + sample_bytes
    if len(payload) != expected:
        raise PacketError("SNIPPET payload_len does not match n_ch/n_samp")
    offset = SNIPPET_PREFIX_SIZE
    if n_ch:
        channel_ids = struct.unpack(f"<{n_ch}H", payload[offset : offset + channel_bytes])
    else:
        channel_ids = ()
    offset += channel_bytes
    samples = np.frombuffer(payload[offset:], dtype="<i2").reshape((n_samp, n_ch)).copy()
    return SnippetPacket(
        event_id=int(event_id),
        t_sample=int(t_sample),
        t0_sample=int(t0_sample),
        sample_rate_hz=float(sample_rate_hz),
        channel_ids=tuple(int(c) for c in channel_ids),
        samples=samples,
        peak_channel=int(peak_channel),
        lsb_uv=float(lsb_uv),
        seq=int(seq),
        header_flags=int(header_flags),
    )
