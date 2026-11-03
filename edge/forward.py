"""Simplified quasi-static monopole extracellular forward model.

This is NOT a biophysical LFPy/MEArec simulator. There is no morphology, no
cable equation, and no medium anisotropy. Each source is a point monopole whose
peak amplitude on an electrode falls as

    V ∝ amplitude / (r_um / 1000 + lambda_mm)

with lambda_mm = 0.08 (80 µm), matching the conceptual length in
``viz.sim.recording_model.RecordingConfig.attenuation_mm``. All geometry stored
and returned by this package is in micrometers.

The temporal kernel is a unit-peak negative triphasic waveform (same formula as
``viz.sim.recording_model.triphasic_template``). Overlapping sources add
linearly; Gaussian noise is added in microvolts.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from edge.types import ElectrodeArray

LAMBDA_UM = 80.0
LAMBDA_MM = LAMBDA_UM / 1000.0
TEMPLATE_DURATION_MS = 2.5
DEFAULT_AMPLITUDE_UV = 12.0


@dataclass(frozen=True)
class PointSource:
    """One monopole event. ``t_sample`` is the sample index of the negative peak."""

    t_sample: int
    xyz_um: tuple[float, float, float]
    amplitude_uv: float = DEFAULT_AMPLITUDE_UV


def triphasic_template(n_samples: int) -> np.ndarray:
    """Unit-peak negative triphasic extracellular-like kernel.

    Copied from ``viz.sim.recording_model.triphasic_template`` without modifying
    that module.
    """
    t = np.linspace(-1.2, 1.3, n_samples, dtype=np.float64)
    waveform = (
        -1.6 * np.exp(-((t) ** 2) / (2 * 0.18**2))
        + 0.55 * np.exp(-((t - 0.45) ** 2) / (2 * 0.22**2))
        + 0.25 * np.exp(-((t + 0.55) ** 2) / (2 * 0.25**2))
    )
    peak = np.min(waveform)
    if peak >= 0:
        raise RuntimeError("template must have a negative peak")
    return waveform / np.abs(peak)


def site_gain_um(
    source_xyz_um: tuple[float, float, float] | np.ndarray,
    electrode_xyz_um: tuple[float, float, float] | np.ndarray,
    lambda_um: float = LAMBDA_UM,
) -> float:
    """Quasi-static gain 1 / (r_mm + lambda_mm) for one source–electrode pair."""
    delta = np.asarray(source_xyz_um, dtype=np.float64).reshape(3) - np.asarray(
        electrode_xyz_um, dtype=np.float64
    ).reshape(3)
    r_um = float(np.linalg.norm(delta))
    return 1.0 / (r_um / 1000.0 + lambda_um / 1000.0)


def predicted_amplitude_patterns(
    xyz_um: np.ndarray,
    electrode_xyz_um: np.ndarray,
    lambda_um: float = LAMBDA_UM,
) -> np.ndarray:
    """Return [n_src, n_channels] quasi-static amplitude gains."""
    sources = np.asarray(xyz_um, dtype=np.float64).reshape(-1, 3)
    electrodes = np.asarray(electrode_xyz_um, dtype=np.float64).reshape(-1, 3)
    delta = sources[:, None, :] - electrodes[None, :, :]
    r_um = np.linalg.norm(delta, axis=2)
    return 1.0 / (r_um / 1000.0 + lambda_um / 1000.0)


def predicted_amplitude_pattern(
    xyz_um: tuple[float, float, float] | np.ndarray,
    electrode_xyz_um: np.ndarray,
    lambda_um: float = LAMBDA_UM,
) -> np.ndarray:
    return predicted_amplitude_patterns(np.asarray(xyz_um, dtype=np.float64), electrode_xyz_um, lambda_um)[0]


def kernel_peak_index(n_kernel: int) -> int:
    kernel = triphasic_template(n_kernel)
    return int(np.argmin(kernel))


def render_recording(
    sources: Sequence[PointSource],
    array: ElectrodeArray,
    sample_rate: float,
    noise_uv: float,
    seed: int,
    n_samples: int | None = None,
    template_duration_ms: float = TEMPLATE_DURATION_MS,
    lambda_um: float = LAMBDA_UM,
) -> np.ndarray:
    """Render [samples, channels] voltages in microvolts.

    ``n_samples`` is optional; when omitted the buffer extends through the last
    source kernel. ``seed`` is passed to ``numpy.random.default_rng``.
    """
    n_channels = array.n_channels
    electrodes = array.xyz_um()
    n_kernel = max(8, int(round(template_duration_ms * sample_rate / 1000.0)))
    kernel = triphasic_template(n_kernel)
    peak_index = int(np.argmin(kernel))
    if n_samples is None:
        if not sources:
            n_samples = n_kernel
        else:
            n_samples = max(int(src.t_sample) + (n_kernel - peak_index) for src in sources)
            n_samples = max(n_samples, max(int(src.t_sample) for src in sources) + 1)
    voltages = np.zeros((int(n_samples), n_channels), dtype=np.float64)

    for source in sources:
        gains = predicted_amplitude_pattern(source.xyz_um, electrodes, lambda_um=lambda_um)
        scaled_by_channel = source.amplitude_uv * gains[:, None] * kernel[None, :]
        center = int(source.t_sample)
        start = center - peak_index
        stop = start + n_kernel
        src_start = 0
        dst_start = start
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        src_stop = n_kernel
        dst_stop = stop
        if dst_stop > n_samples:
            src_stop -= dst_stop - n_samples
            dst_stop = n_samples
        if dst_stop <= dst_start or src_stop <= src_start:
            continue
        voltages[dst_start:dst_stop, :] += scaled_by_channel[:, src_start:src_stop].T

    if noise_uv > 0:
        rng = np.random.default_rng(seed)
        voltages += rng.normal(0.0, noise_uv, size=voltages.shape)
    if not np.isfinite(voltages).all():
        raise ValueError("forward model produced non-finite samples")
    return voltages
