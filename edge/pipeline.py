"""Receive ASIC events, optionally localize, bin by y, and decode.

Packets are decoded only if ``edge.link.codec`` is importable. Otherwise callers
pass typed objects (Snippet, LocalizationEstimate, or StreamEvent).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def _snippet_from_packet(packet) -> Snippet:
    from edge.geometry import linear_probe_um

    lsb = float(getattr(packet, "lsb_uv", 1.0))
    voltages = np.asarray(packet.samples, dtype=np.float64) * lsb
    channel_ids = tuple(int(c) for c in packet.channel_ids)
    array = linear_probe_um(n_channels=max(len(channel_ids), 1))
    xyz = array.xyz_um()
    if len(channel_ids) != xyz.shape[0]:
        order = np.array(channel_ids, dtype=np.int64)
        xyz = linear_probe_um(n_channels=int(np.max(order)) + 1).xyz_um()[order]
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


def ingest(items: Sequence[object]) -> list[StreamEvent]:
    """Normalize packets or typed objects into StreamEvent records."""
    events: list[StreamEvent] = []
    pending_snippet: dict[int, Snippet] = {}
    pending_loc: dict[int, LocalizationEstimate] = {}

    def flush_id(event_id: int, t_sample: int, channel_id: int) -> None:
        events.append(
            StreamEvent(
                event_id=event_id,
                t_sample=t_sample,
                channel_id=channel_id,
                snippet=pending_snippet.pop(event_id, None),
                localization=pending_loc.pop(event_id, None),
            )
        )

    event_packet_type = None
    snippet_packet_type = None
    try:
        from edge.link.codec import EventPacket as event_packet_type
        from edge.link.codec import SnippetPacket as snippet_packet_type
    except ImportError:
        pass

    for item in items:
        if isinstance(item, (bytes, bytearray)):
            decoded = _try_codec_decode(bytes(item))
            if decoded is None:
                raise TypeError(
                    "bytes packet supplied but edge.link.codec is not importable; "
                    "pass Snippet / StreamEvent objects instead"
                )
            item = decoded
        if isinstance(item, StreamEvent):
            events.append(item)
        elif snippet_packet_type is not None and isinstance(item, snippet_packet_type):
            snippet = _snippet_from_packet(item)
            pending_snippet[snippet.event_id] = snippet
            flush_id(snippet.event_id, snippet.t_sample, int(snippet.peak_channel or 0))
        elif event_packet_type is not None and isinstance(item, event_packet_type):
            flush_id(int(item.event_id), int(item.t_sample), int(item.channel_id))
        elif isinstance(item, Snippet):
            channel = item.peak_channel if item.peak_channel is not None else 0
            pending_snippet[item.event_id] = item
            flush_id(item.event_id, item.t_sample, int(channel))
        elif isinstance(item, LocalizationEstimate):
            if item.event_id is None:
                continue
            pending_loc[item.event_id] = item
            if events and events[-1].event_id == item.event_id and events[-1].localization is None:
                events[-1] = replace(events[-1], localization=item)
        elif isinstance(item, tuple) and len(item) == 2:
            ingest([item[0]])
            extra = ingest([item[1]])
            events.extend(extra)
        else:
            raise TypeError(f"unsupported pipeline input: {type(item)!r}")
    return events


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
