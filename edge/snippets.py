"""Extract event-aligned multichannel snippets from a voltage recording."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from edge import SAMPLE_RATE_HZ
from edge.types import ElectrodeArray, Snippet, snippet_sample_counts


def extract_snippet(
    voltages_uv: np.ndarray,
    t_sample: int,
    array: ElectrodeArray,
    *,
    event_id: int = 0,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    n_pre: int | None = None,
    n_post: int | None = None,
    ground_truth_xyz_um: tuple[float, float, float] | None = None,
    peak_channel: int | None = None,
) -> Snippet:
    """Window ``[t_sample - n_pre, t_sample + n_post)`` with zero padding.

    ``t_sample - t0_sample`` equals ``n_pre``. The alignment sample inside the
    window is therefore index ``n_pre``.
    """
    samples = np.asarray(voltages_uv, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("voltages_uv must be [samples, channels]")
    if n_pre is None or n_post is None:
        default_pre, default_post = snippet_sample_counts(sample_rate_hz)
        n_pre = default_pre if n_pre is None else n_pre
        n_post = default_post if n_post is None else n_post
    n_pre = int(n_pre)
    n_post = int(n_post)
    t_sample = int(t_sample)
    t0_sample = t_sample - n_pre
    n_window = n_pre + n_post
    n_rec, n_ch = samples.shape
    if n_ch != array.n_channels:
        raise ValueError("recording channel count does not match electrode array")

    window = np.zeros((n_window, n_ch), dtype=np.float64)
    src_start = t0_sample
    dst_start = 0
    if src_start < 0:
        dst_start = -src_start
        src_start = 0
    src_stop = t0_sample + n_window
    dst_stop = n_window
    if src_stop > n_rec:
        dst_stop -= src_stop - n_rec
        src_stop = n_rec
    if dst_stop > dst_start and src_stop > src_start:
        window[dst_start:dst_stop] = samples[src_start:src_stop]

    channel_ids = array.channel_ids()
    if peak_channel is None:
        peak_channel = _peak_channel_id(window, n_pre, channel_ids)

    return Snippet(
        event_id=event_id,
        t_sample=t_sample,
        t0_sample=t0_sample,
        sample_rate_hz=float(sample_rate_hz),
        channel_ids=channel_ids,
        voltages_uv=window,
        electrode_xyz_um=np.asarray(array.xyz_um(), dtype=np.float64),
        peak_channel=peak_channel,
        ground_truth_xyz_um=ground_truth_xyz_um,
    )


def extract_snippets(
    voltages_uv: np.ndarray,
    t_samples: Sequence[int],
    array: ElectrodeArray,
    **kwargs: object,
) -> list[Snippet]:
    ground_truths = kwargs.pop("ground_truth_xyz_um", None)
    snippets = []
    for index, t_sample in enumerate(t_samples):
        gt = None
        if ground_truths is not None:
            gt = ground_truths[index]
        snippets.append(
            extract_snippet(
                voltages_uv,
                int(t_sample),
                array,
                event_id=index,
                ground_truth_xyz_um=gt,
                **kwargs,
            )
        )
    return snippets


def channel_peak_amplitudes(snippet: Snippet) -> np.ndarray:
    """Positive per-channel spike magnitude from the most negative sample."""
    mins = np.min(snippet.voltages_uv, axis=0)
    amps = np.maximum(-mins, 0.0)
    if float(np.max(amps)) <= 0.0:
        return np.max(np.abs(snippet.voltages_uv), axis=0)
    return amps


def channel_rms_amplitudes(snippet: Snippet) -> np.ndarray:
    return np.sqrt(np.mean(np.square(snippet.voltages_uv), axis=0))


def _peak_channel_id(
    window: np.ndarray, n_pre: int, channel_ids: tuple[int, ...]
) -> int:
    if 0 <= n_pre < window.shape[0]:
        idx = int(np.argmin(window[n_pre]))
    else:
        idx = int(np.argmin(np.min(window, axis=0)))
    return int(channel_ids[idx])
