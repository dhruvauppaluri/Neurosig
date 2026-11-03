"""Receive ASIC events, optionally localize, bin by y, and decode.

Packets are decoded only if ``edge.link.codec`` is importable. Otherwise callers
pass typed objects (Snippet, LocalizationEstimate, or StreamEvent).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Iterable, Sequence

import numpy as np

from edge import N_CHANNELS, SAMPLE_RATE_HZ
from edge.decode.features import N_Y_BINS, Y_MAX_UM, Y_MIN_UM, spatial_vector, y_occupancy
from edge.decode.model import logistic_confidence
from edge.geometry import linear_probe_um
from edge.localize.base import LocalizationEngine, NotImplementedLocalizer
from edge.types import LocalizationEstimate, Snippet


@dataclass(frozen=True)
class StreamEvent:
    event_id: int
    t_sample: int
    channel_id: int
    snippet: Snippet | None = None
    localization: LocalizationEstimate | None = None


@dataclass(frozen=True)
class DecodePrediction:
    t_sample: int
    t_s: float
    label: int
    score: float
    confidence: float
    n_events: int
    occupancy: tuple[float, ...]
    used_localization: bool
    reason: str


def _try_codec_decode(payload: bytes):
    try:
        from edge.link.codec import decode_packet
    except ImportError:
        return None
    return decode_packet(payload)


def _snippet_from_packet(packet, electrode_positions_um: dict[int, tuple[float, float, float]] | None = None) -> Snippet:
    from edge.geometry import linear_probe_um

    lsb = float(getattr(packet, "lsb_uv", 1.0))
    voltages = np.asarray(packet.samples, dtype=np.float64) * lsb
    channel_ids = tuple(int(c) for c in packet.channel_ids)
    if electrode_positions_um is not None:
        xyz = np.asarray([electrode_positions_um[channel] for channel in channel_ids], dtype=np.float64)
    else:
        xyz = linear_probe_um(n_channels=int(max(channel_ids)) + 1).xyz_um()[list(channel_ids)]
    peak = int(packet.peak_channel)
    return Snippet(
        event_id=int(packet.event_id),
        t_sample=int(packet.t_sample),
        t0_sample=int(packet.t0_sample),
        sample_rate_hz=float(packet.sample_rate_hz),
        channel_ids=channel_ids,
        voltages_uv=voltages,
        electrode_xyz_um=np.asarray(xyz, dtype=np.float64),
        peak_channel=peak,
    )


def _import_localizer() -> LocalizationEngine | None:
    import importlib
    import pkgutil

    try:
        from edge.localize.amplitude import AmplitudeLocalizer
        return AmplitudeLocalizer()
    except (ImportError, TypeError):
        pass

    import edge.localize as pkg

    for _finder, name, _ispkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if name.rsplit(".", 1)[-1] in {"base"}:
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for attr in vars(mod).values():
            if not isinstance(attr, type):
                continue
            if attr is NotImplementedLocalizer or attr.__name__ == "NotImplementedLocalizer":
                continue
            if not hasattr(attr, "estimate"):
                continue
            try:
                inst = attr()
            except TypeError:
                continue
            if isinstance(inst, NotImplementedLocalizer):
                continue
            if hasattr(inst, "estimate"):
                return inst  # type: ignore[return-value]
    return None


@dataclass
class IngestDiagnostics:
    duplicate_packets: int = 0
    corrupt_packets: int = 0
    missing_packets: int = 0
    missing_snippets: int = 0
    snippet_only: int = 0
    ignored_control_packets: int = 0


@dataclass
class _PendingEvent:
    event: object | None = None
    snippet: Snippet | None = None
    localization: LocalizationEstimate | None = None
    stream: StreamEvent | None = None


class EventAssembler:
    """Assemble a finite NSG1 session; call finish at its end-of-stream boundary.

    IDs are scoped to this instance/session. Duplicate IDs never become extra spikes.
    max_events bounds memory while allowing packets to arrive in either order.
    """

    def __init__(self, *, session_id: str = "default", max_events: int = 100_000,
                 corrupt: str = "raise",
                 electrode_positions_um: dict[int, tuple[float, float, float]] | None = None) -> None:
        if max_events < 1 or corrupt not in {"raise", "skip"}:
            raise ValueError("invalid assembler limits or corruption policy")
        self.session_id = session_id
        self.max_events = max_events
        self.corrupt = corrupt
        self.electrode_positions_um = electrode_positions_um
        self.diagnostics = IngestDiagnostics()
        self._pending: dict[int, _PendingEvent] = {}
        self._seq: set[int] = set()
        self._seq_digest: dict[int, bytes] = {}
        self._closed = False

    def _slot(self, event_id: int) -> _PendingEvent:
        if event_id not in self._pending:
            if len(self._pending) >= self.max_events:
                raise OverflowError("session event capacity exceeded; finish or split the recording")
            self._pending[event_id] = _PendingEvent()
        return self._pending[event_id]

    def feed(self, item: object) -> None:
        from edge.link.codec import (AckPacket, CmdPacket, EventPacket, PacketError,
                                     SnippetPacket, decode_packet, encode_packet)

        if self._closed:
            raise RuntimeError("assembler already finished")
        if isinstance(item, tuple) and len(item) == 2:
            self.feed(item[0]); self.feed(item[1])
            return
        if isinstance(item, (bytes, bytearray, memoryview)):
            try:
                item = decode_packet(bytes(item))
            except PacketError:
                self.diagnostics.corrupt_packets += 1
                if self.corrupt == "raise":
                    raise
                return
        if isinstance(item, (CmdPacket, AckPacket)):
            self.diagnostics.ignored_control_packets += 1
            return
        if isinstance(item, (EventPacket, SnippetPacket)):
            seq = int(item.seq) & 0xFFFFFFFF
            digest = hashlib.sha256(encode_packet(item)).digest()
            if seq in self._seq_digest and self._seq_digest[seq] != digest:
                raise ValueError(f"conflicting packet sequence {seq}")
            self._seq_digest[seq] = digest
            self._seq.add(seq)
        if isinstance(item, EventPacket):
            slot = self._slot(int(item.event_id))
            if slot.event is not None:
                old = slot.event
                if not isinstance(old, EventPacket) or (old.event_id, old.t_sample, old.channel_id,
                    old.amplitude, old.flags) != (item.event_id, item.t_sample, item.channel_id,
                    item.amplitude, item.flags):
                    raise ValueError(f"conflicting EVENT for ID {item.event_id}")
                self.diagnostics.duplicate_packets += 1
            else:
                slot.event = item
        elif isinstance(item, SnippetPacket):
            self._add_snippet(_snippet_from_packet(item, self.electrode_positions_um))
        elif isinstance(item, Snippet):
            self._add_snippet(item)
        elif isinstance(item, StreamEvent):
            slot = self._slot(item.event_id)
            if slot.stream is not None:
                old = slot.stream
                same_snippet = old.snippet is item.snippet
                if old.snippet is not None and item.snippet is not None:
                    same_snippet = (old.snippet.t_sample == item.snippet.t_sample
                                    and old.snippet.channel_ids == item.snippet.channel_ids
                                    and np.array_equal(old.snippet.voltages_uv, item.snippet.voltages_uv))
                if not (old.t_sample == item.t_sample and old.channel_id == item.channel_id
                        and same_snippet and old.localization == item.localization):
                    raise ValueError(f"conflicting StreamEvent for ID {item.event_id}")
            if slot.stream is not None:
                self.diagnostics.duplicate_packets += 1
            slot.stream = item
        elif isinstance(item, LocalizationEstimate):
            if item.event_id is None:
                return
            slot = self._slot(item.event_id)
            if slot.localization is not None and slot.localization != item:
                raise ValueError(f"conflicting localization for ID {item.event_id}")
            slot.localization = item
        else:
            raise TypeError(f"unsupported pipeline input: {type(item)!r}")

    def _add_snippet(self, snippet: Snippet) -> None:
        slot = self._slot(snippet.event_id)
        if slot.snippet is not None:
            old = slot.snippet
            same = (old.t_sample == snippet.t_sample and old.t0_sample == snippet.t0_sample
                    and old.channel_ids == snippet.channel_ids and old.peak_channel == snippet.peak_channel
                    and old.sample_rate_hz == snippet.sample_rate_hz
                    and np.array_equal(old.voltages_uv, snippet.voltages_uv))
            if not same:
                raise ValueError(f"conflicting SNIPPET for ID {snippet.event_id}")
            self.diagnostics.duplicate_packets += 1
        else:
            slot.snippet = snippet

    def finish(self) -> list[StreamEvent]:
        from edge.link.codec import EventPacket
        from edge.link.schema import FLAG_HAS_SNIPPET

        if self._closed:
            raise RuntimeError("assembler already finished")
        self._closed = True
        if len(self._seq) > 1:
            # Exclude the largest circular gap: the remaining shortest span
            # handles reordered arrivals and 32-bit wrap without false loss.
            ordered = sorted(self._seq)
            gaps = [((ordered[(i + 1) % len(ordered)] - seq) & 0xFFFFFFFF)
                    for i, seq in enumerate(ordered)]
            span = (1 << 32) - max(gaps) + 1
            self.diagnostics.missing_packets = span - len(ordered)
        result: list[StreamEvent] = []
        for event_id, slot in self._pending.items():
            if slot.stream is not None:
                if slot.event is not None or slot.snippet is not None:
                    raise ValueError(f"mixed StreamEvent and packet inputs for ID {event_id}")
                result.append(replace(slot.stream, localization=slot.localization or slot.stream.localization))
                continue
            event = slot.event
            snippet = slot.snippet
            if event is None and snippet is None:
                continue
            if isinstance(event, EventPacket):
                if snippet is not None and (event.t_sample != snippet.t_sample or
                                            event.channel_id != snippet.peak_channel):
                    raise ValueError(f"EVENT/SNIPPET metadata mismatch for ID {event_id}")
                if snippet is None and event.flags & FLAG_HAS_SNIPPET:
                    self.diagnostics.missing_snippets += 1
                t_sample, channel_id = event.t_sample, event.channel_id
            else:
                assert snippet is not None
                self.diagnostics.snippet_only += 1
                t_sample = snippet.t_sample
                channel_id = int(snippet.peak_channel or 0)
            result.append(StreamEvent(event_id, int(t_sample), int(channel_id), snippet, slot.localization))
        return result


def ingest(items: Sequence[object], *, session_id: str = "default",
           max_events: int = 100_000, corrupt: str = "raise") -> list[StreamEvent]:
    """Normalize one finite session into exactly one StreamEvent per event ID."""
    assembler = EventAssembler(session_id=session_id, max_events=max_events, corrupt=corrupt)
    for item in items:
        assembler.feed(item)
    return assembler.finish()


def localize_events(
    events: Sequence[StreamEvent],
    engine: LocalizationEngine | None = None,
) -> list[StreamEvent]:
    resolved = engine if engine is not None else _import_localizer()
    if resolved is None or isinstance(resolved, NotImplementedLocalizer):
        return list(events)
    out: list[StreamEvent] = []
    for event in events:
        if event.localization is not None or event.snippet is None:
            out.append(event)
            continue
        estimate = resolved.estimate(event.snippet)
        out.append(replace(event, localization=estimate))
    return out


def _event_y_um(event: StreamEvent, probe_y: np.ndarray) -> tuple[float | None, bool]:
    loc = event.localization
    if loc is not None and loc.available and loc.xyz_um is not None:
        return float(loc.xyz_um[1]), True
    if 0 <= event.channel_id < len(probe_y):
        return float(probe_y[event.channel_id]), False
    return None, False


def bin_by_y_um(
    events: Sequence[StreamEvent],
    n_bins: int = N_Y_BINS,
    y_min: float = Y_MIN_UM,
    y_max: float = Y_MAX_UM,
) -> np.ndarray:
    probe_y = linear_probe_um(n_channels=N_CHANNELS).xyz_um()[:, 1]
    ys = []
    for event in events:
        y_um, _used = _event_y_um(event, probe_y)
        if y_um is not None:
            ys.append(y_um)
    return y_occupancy(np.asarray(ys, dtype=np.float64), n_bins=n_bins, y_min=y_min, y_max=y_max)


def _window_features(events: Sequence[StreamEvent]) -> tuple[np.ndarray, bool]:
    counts = np.zeros(N_CHANNELS, dtype=np.float64)
    xyz_points: list[tuple[float, float, float]] = []
    used_loc = False
    probe_y = linear_probe_um(n_channels=N_CHANNELS).xyz_um()[:, 1]
    for event in events:
        if 0 <= event.channel_id < N_CHANNELS:
            counts[event.channel_id] += 1.0
        loc = event.localization
        if loc is not None and loc.available and loc.xyz_um is not None:
            xyz_points.append(loc.xyz_um)
            used_loc = True
        elif 0 <= event.channel_id < len(probe_y):
            xyz_points.append((0.0, float(probe_y[event.channel_id]), 0.0))
    counts = np.clip(counts, 0, 255)
    features = spatial_vector(counts, np.asarray(xyz_points, dtype=np.float64) if xyz_points else np.zeros((0, 3)))
    return features, used_loc


def decode_windows(
    events: Sequence[StreamEvent],
    weights: np.ndarray,
    bias: float,
    *,
    window_samples: int = int(0.1 * SAMPLE_RATE_HZ),
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> list[DecodePrediction]:
    """Bin events into time windows and emit a logistic prediction per window."""
    if not events:
        return []
    weights = np.asarray(weights, dtype=np.float64)
    ordered = sorted(events, key=lambda e: (e.t_sample, e.event_id))
    t0 = ordered[0].t_sample
    buckets: dict[int, list[StreamEvent]] = {}
    for event in ordered:
        idx = (event.t_sample - t0) // window_samples
        buckets.setdefault(int(idx), []).append(event)

    predictions: list[DecodePrediction] = []
    for idx in sorted(buckets):
        group = buckets[idx]
        features, used_loc = _window_features(group)
        if features.shape[0] != weights.shape[0]:
            # Baseline 32-weight models score counts only.
            score_vec = features[: weights.shape[0]]
        else:
            score_vec = features
        score = float(score_vec @ weights + float(bias))
        confidence = logistic_confidence(score)
        label = int(score >= 0)
        t_sample = t0 + idx * window_samples
        predictions.append(
            DecodePrediction(
                t_sample=int(t_sample),
                t_s=float(t_sample) / float(sample_rate_hz),
                label=label,
                score=score,
                confidence=confidence,
                n_events=len(group),
                occupancy=tuple(float(v) for v in bin_by_y_um(group)),
                used_localization=used_loc,
                reason=(
                    "localized y_um bins"
                    if used_loc
                    else "localizer skipped; occupancy from channel geometry y_um"
                ),
            )
        )
    return predictions


def run_pipeline(
    items: Sequence[object] | Iterable[object],
    weights: np.ndarray,
    bias: float,
    *,
    localizer: LocalizationEngine | None = None,
    window_samples: int = int(0.1 * SAMPLE_RATE_HZ),
) -> list[DecodePrediction]:
    events = ingest(list(items))
    events = localize_events(events, engine=localizer)
    return decode_windows(events, weights, bias, window_samples=window_samples)
