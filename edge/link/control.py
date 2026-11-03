"""Software-only control path for the emulated ASIC.

Supported opcodes update detector thresholds, the channel enable mask, and the
acquisition mode, and they report buffer/link status or answer ping/diag.
There is no stimulation command and this module never emits a waveform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edge import N_CHANNELS
from edge.link.buffer import LinkFifo
from edge.link.codec import AckPacket, CmdPacket
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
    DEFAULT_FIFO_DEPTH,
    SCHEMA_VERSION,
)

STATUS_OK = 0
STATUS_BAD_OPCODE = 1
STATUS_BAD_ARGS = 2
STATUS_BAD_MODE = 3

I16_MIN = -32768
I16_MAX = 32767

ACQ_MODES = (ACQ_EVENT, ACQ_SNIPPET, ACQ_BOTH)


def _clip_i16(value: int) -> int:
    if value < I16_MIN or value > I16_MAX:
        raise ValueError("threshold must fit in signed int16")
    return int(value)


@dataclass
class ControlPlane:
    """Holds runtime configuration the software ASIC model can consult."""

    n_channels: int = N_CHANNELS
    mode: int = ACQ_BOTH
    fifo: LinkFifo | None = None
    thresholds: list[int] = field(default_factory=list)
    enabled: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if not self.thresholds:
            self.thresholds = [0] * self.n_channels
        elif len(self.thresholds) != self.n_channels:
            raise ValueError("thresholds length must equal n_channels")
        self.thresholds = [_clip_i16(v) for v in self.thresholds]
        if not self.enabled:
            self.enabled = [True] * self.n_channels
        elif len(self.enabled) != self.n_channels:
            raise ValueError("enabled length must equal n_channels")
        self.enabled = [bool(v) for v in self.enabled]
        if self.mode not in ACQ_MODES:
            raise ValueError("unknown acquisition mode")

    def channel_enabled(self, channel_id: int) -> bool:
        if 0 <= channel_id < self.n_channels:
            return self.enabled[channel_id]
        return False

    def apply(self, cmd: CmdPacket) -> AckPacket:
        """Execute one command and return an ACK with the same sequence number."""
        if cmd.opcode == CMD_SET_THRESHOLD:
            return self._ack(cmd, *self._set_threshold(cmd.args))
        if cmd.opcode == CMD_SET_CHANNELS:
            return self._ack(cmd, *self._set_channels(cmd.args))
        if cmd.opcode == CMD_SET_MODE:
            return self._ack(cmd, *self._set_mode(cmd.args))
        if cmd.opcode == CMD_GET_STATUS:
            return self._ack(cmd, STATUS_OK, self._status_args())
        if cmd.opcode == CMD_PING:
            return self._ack(cmd, STATUS_OK, cmd.args)
        if cmd.opcode == CMD_DIAG:
            return self._ack(cmd, STATUS_OK, self._diag_args())
        return self._ack(cmd, STATUS_BAD_OPCODE, ())

    def _ack(self, cmd: CmdPacket, status: int, args: tuple[int, ...]) -> AckPacket:
        return AckPacket(opcode=cmd.opcode, status=status, args=tuple(int(a) for a in args), seq=cmd.seq, header_flags=cmd.header_flags)

    def _set_threshold(self, args: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if len(args) == 1:
            value = _clip_i16(args[0])
            self.thresholds = [value] * self.n_channels
            return STATUS_OK, (value,)
        if len(args) == 2:
            channel, value = int(args[0]), _clip_i16(args[1])
            if channel < 0 or channel >= self.n_channels:
                return STATUS_BAD_ARGS, ()
            self.thresholds[channel] = value
            return STATUS_OK, (channel, value)
        if len(args) == self.n_channels:
            self.thresholds = [_clip_i16(v) for v in args]
            return STATUS_OK, tuple(self.thresholds)
        return STATUS_BAD_ARGS, ()

    def _set_channels(self, args: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if not args:
            return STATUS_BAD_ARGS, ()
        if len(args) == self.n_channels and all(a in (0, 1) for a in args):
            self.enabled = [bool(a) for a in args]
            return STATUS_OK, tuple(int(v) for v in self.enabled)
        n_words = (self.n_channels + 31) // 32
        if len(args) != n_words:
            return STATUS_BAD_ARGS, ()
        enabled = [False] * self.n_channels
        for word_index, raw in enumerate(args):
            bits = int(raw) & 0xFFFFFFFF
            for bit in range(32):
                channel = word_index * 32 + bit
                if channel >= self.n_channels:
                    break
                enabled[channel] = bool((bits >> bit) & 1)
        self.enabled = enabled
        return STATUS_OK, tuple(int(v) for v in self.enabled)

    def _set_mode(self, args: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if len(args) != 1:
            return STATUS_BAD_ARGS, ()
        mode = int(args[0])
        if mode not in ACQ_MODES:
            return STATUS_BAD_MODE, ()
        self.mode = mode
        return STATUS_OK, (mode,)

    def _status_args(self) -> tuple[int, ...]:
        fifo = self.fifo
        fifo_args = fifo.status_args() if fifo is not None else (0, DEFAULT_FIFO_DEPTH, 0, 0, 0, 0, 0, 0, 0)
        return (
            self.mode,
            self.n_channels,
            sum(1 for bit in self.enabled if bit),
            *fifo_args,
        )

    def _diag_args(self) -> tuple[int, ...]:
        # Diagnostic ping of software identity. No stimulation payload.
        return (SCHEMA_VERSION, self.n_channels, self.mode, 0)
