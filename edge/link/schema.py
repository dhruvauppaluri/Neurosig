"""Versioned little-endian NeuroSig packet layouts.

Byte order is little-endian. Signed fields are two's complement. CRC32 covers
every byte before the CRC field (IEEE 802.3 polynomial, ``binascii.crc32``).
An RTL packetizer may emit these bytes later; this module does not implement RTL.
"""

from __future__ import annotations

import struct

MAGIC = b"NSG1"
SCHEMA_VERSION = 1

PKT_EVENT = 0
PKT_SNIPPET = 1
PKT_CMD = 2
PKT_ACK = 3

CMD_SET_THRESHOLD = 1
CMD_SET_CHANNELS = 2
CMD_SET_MODE = 3
CMD_GET_STATUS = 4
CMD_DIAG = 5
CMD_PING = 6

ACQ_EVENT = 0
ACQ_SNIPPET = 1
ACQ_BOTH = 2

# Header: magic[4] version u16 type u8 flags u8 seq u32 payload_len u32
HEADER_STRUCT = struct.Struct("<4sHBBII")
HEADER_SIZE = HEADER_STRUCT.size  # 16
CRC_SIZE = 4

# EVENT payload: event_id u32, t_sample u64, channel_id u16, amplitude i16, flags u16, reserved u16
EVENT_STRUCT = struct.Struct("<IQHhHH")
EVENT_PAYLOAD_SIZE = EVENT_STRUCT.size  # 20

# SNIPPET prefix then channel_ids[n_ch] as u16 then samples as int16[n_samp * n_ch] sample-major.
# event_id u32, t_sample u64, t0_sample u64, sample_rate_hz f32, n_ch u16, n_samp u16,
# peak_channel i16, lsb_uv f32
SNIPPET_PREFIX_STRUCT = struct.Struct("<IQQfHHhf")
SNIPPET_PREFIX_SIZE = SNIPPET_PREFIX_STRUCT.size  # 34

# CMD/ACK: opcode u16, status u16, n_args u16, reserved u16, then n_args little-endian int32
CMD_PREFIX_STRUCT = struct.Struct("<HHHH")
CMD_PREFIX_SIZE = CMD_PREFIX_STRUCT.size

FLAG_HAS_SNIPPET = 0x0001
FLAG_OVERFLOW = 0x0002
FLAG_FLUSH = 0x0004

DEFAULT_FIFO_DEPTH = 64
DROP_OLDEST = True
