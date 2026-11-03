"""Software stand-in for an ASIC that emits EVENT and SNIPPET packets.

Detector ``Event`` objects (``scripts.detect_spikes.Event`` or anything with
``sample``, ``channel``, and ``voltage_uv``) plus a voltage window become
little-endian NeuroSig frames. ``t_sample`` is the recording sample index of
the event peak and is copied onto both packets. In the snippet array the peak
row is ``t_sample - t0_sample``.

Voltages are quantized to signed int16 with LSB ``lsb_uv``. The default matches
``scripts.quantize_adc.AdcConfig.lsb_uv`` (16-bit, ±135 µV). This module does
not modify DSP scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge import N_CHANNELS, SAMPLE_RATE_HZ
from edge.types import snippet_sample_counts
from edge.link.buffer import LinkFifo
from edge.link.codec import EventPacket, Packet, SnippetPacket, encode_packet
from edge.link.control import ControlPlane
from edge.link.schema import (
    ACQ_BOTH,
    ACQ_EVENT,
    ACQ_SNIPPET,
    FLAG_HAS_SNIPPET,
    FLAG_OVERFLOW,
)
from scripts.quantize_adc import AdcConfig, quantize_voltages


def default_lsb_uv() -> float:
    """Documented snippet/event LSB in microvolts (``AdcConfig`` default)."""
    return float(AdcConfig().lsb_uv)


def quantize_uv(voltages_uv: np.ndarray, lsb_uv: float | None = None) -> np.ndarray:
    """Quantize microvolts to int16 codes using ``AdcConfig`` rounding/saturation."""
    config = AdcConfig() if lsb_uv is None else _config_for_lsb(lsb_uv)
    codes, _saturated = quantize_voltages(voltages_uv, config)
    return codes


def _config_for_lsb(lsb_uv: float) -> AdcConfig:
    """Build an ``AdcConfig`` whose ``lsb_uv`` matches ``lsb_uv`` at 16 bits.

    ``AdcConfig.lsb_uv = 2 * full_scale_uv / 2**bits``, so
    ``full_scale_uv = lsb_uv * 2**(bits-1)``.
    """
    bits = 16
    return AdcConfig(bits=bits, full_scale_uv=float(lsb_uv) * (1 << (bits - 1)))


@dataclass
class AsicModel:
    n_channels: int = N_CHANNELS
    sample_rate_hz: float = SAMPLE_RATE_HZ
    lsb_uv: float | None = None
    n_pre: int | None = None
    n_post: int | None = None
    control: ControlPlane | None = None
    fifo: LinkFifo | None = None
    next_event_id: int = 1
    next_seq: int = 0

    def __post_init__(self) -> None:
        if self.lsb_uv is None:
            self.lsb_uv = default_lsb_uv()
        if self.lsb_uv <= 0:
            raise ValueError("lsb_uv must be positive")
        if self.n_pre is None or self.n_post is None:
            n_pre, n_post = snippet_sample_counts(self.sample_rate_hz)
            if self.n_pre is None:
                self.n_pre = n_pre
            if self.n_post is None:
                self.n_post = n_post
        if self.control is None:
            self.control = ControlPlane(n_channels=self.n_channels, fifo=self.fifo)
        elif self.fifo is not None and self.control.fifo is None:
            self.control.fifo = self.fifo

    @property
    def adc_config(self) -> AdcConfig:
        return _config_for_lsb(self.lsb_uv)

    def packets_for_event(
        self,
        event: object,
        voltages_uv: np.ndarray,
        channel_ids: tuple[int, ...] | None = None,
        event_id: int | None = None,
    ) -> list[Packet]:
        """Wrap one detector event and its voltage window as EVENT and/or SNIPPET packets."""
        t_sample = int(getattr(event, "sample"))
        channel_id = int(getattr(event, "channel"))
        voltage_uv = float(getattr(event, "voltage_uv"))
        samples = np.asarray(voltages_uv, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("voltages_uv must be [samples, channels]")
        n_time, n_ch_data = samples.shape
        ids = tuple(range(n_ch_data) if channel_ids is None else channel_ids)
        if len(ids) != n_ch_data:
            raise ValueError("channel_ids length must match voltage columns")
        if self.control is not None and not self.control.channel_enabled(channel_id):
            return []

        eid = self.next_event_id if event_id is None else int(event_id)
        if event_id is None:
            self.next_event_id = eid + 1

        mode = self.control.mode if self.control is not None else ACQ_BOTH
        want_event = mode in (ACQ_EVENT, ACQ_BOTH)
        want_snippet = mode in (ACQ_SNIPPET, ACQ_BOTH)
        event_flags = FLAG_HAS_SNIPPET if want_event and want_snippet else 0
        if self.fifo is not None and self.fifo.full:
            event_flags |= FLAG_OVERFLOW

        amplitude = int(quantize_uv(np.asarray([voltage_uv]), self.lsb_uv).reshape(-1)[0])
        packets: list[Packet] = []
        if want_event:
            packets.append(
                EventPacket(
                    event_id=eid,
                    t_sample=t_sample,
                    channel_id=channel_id,
                    amplitude=amplitude,
                    flags=event_flags,
                    seq=self._take_seq(),
                )
            )
        if want_snippet:
            t0_sample = t_sample - int(self.n_pre)
            n_samp = int(self.n_pre) + int(self.n_post)
            window = np.zeros((n_samp, n_ch_data), dtype=np.float64)
            src_start = max(0, t0_sample)
            src_end = min(n_time, t0_sample + n_samp)
            dst_start = src_start - t0_sample
            dst_end = dst_start + (src_end - src_start)
            if src_end > src_start:
                window[dst_start:dst_end] = samples[src_start:src_end]
            codes = quantize_uv(window, self.lsb_uv)
            packets.append(
                SnippetPacket(
                    event_id=eid,
                    t_sample=t_sample,
                    t0_sample=t0_sample,
                    sample_rate_hz=float(self.sample_rate_hz),
                    channel_ids=ids,
                    samples=codes,
                    peak_channel=channel_id,
                    lsb_uv=float(self.lsb_uv),
                    seq=self._take_seq(),
                )
            )
        return packets

    def emit(self, event: object, voltages_uv: np.ndarray, **kwargs: object) -> list[bytes]:
        """Encode packets and optionally push them into the link FIFO."""
        frames = [encode_packet(pkt) for pkt in self.packets_for_event(event, voltages_uv, **kwargs)]  # type: ignore[arg-type]
        if self.fifo is not None:
            for frame in frames:
                self.fifo.try_push(frame)
        return frames

    def _take_seq(self) -> int:
        seq = int(self.next_seq) & 0xFFFFFFFF
        self.next_seq = (seq + 1) & 0xFFFFFFFF
        return seq
