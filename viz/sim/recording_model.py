"""Convert synthetic source events into a multichannel extracellular voltage snippet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viz.sim.events import NeuralEvent
from viz.sim.localization import ElectrodeArray, ElectrodeSite, default_probe_array


@dataclass(frozen=True)
class RecordingConfig:
    sample_rate_hz: float = 32_000.0
    noise_uv: float = 8.0
    attenuation_mm: float = 0.08
    template_duration_ms: float = 2.5
    seed: int = 7


def triphasic_template(n_samples: int) -> np.ndarray:
    """Unit-peak negative triphasic extracellular-like kernel."""
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


def site_gain(source_xyz: tuple[float, float, float], site: ElectrodeSite, length_mm: float) -> float:
    delta = np.array(source_xyz) - np.array(site.xyz_mm)
    distance = float(np.linalg.norm(delta))
    return 1.0 / (distance + length_mm)


def render_voltages(
    events: list[NeuralEvent],
    window_start_s: float,
    window_duration_s: float,
    electrodes: ElectrodeArray | None = None,
    config: RecordingConfig | None = None,
) -> tuple[np.ndarray, ElectrodeArray]:
    """Return finite [samples, channels] voltages in microvolts.

    Overlapping events superimpose. This is a toy point-source model.
    """
    cfg = config or RecordingConfig()
    array = electrodes or default_probe_array()
    n_samples = int(round(window_duration_s * cfg.sample_rate_hz))
    n_channels = len(array.sites)
    voltages = np.zeros((n_samples, n_channels), dtype=np.float64)
    kernel_n = max(8, int(round(cfg.template_duration_ms * cfg.sample_rate_hz / 1000.0)))
    kernel = triphasic_template(kernel_n)
    peak_index = int(np.argmin(kernel))
    rng = np.random.default_rng(cfg.seed + int(window_start_s * 1_000_000) % 10_000_000)

    for event in events:
        if event.kind != "synthetic_source":
            continue
        center = int(round((event.t_s - window_start_s) * cfg.sample_rate_hz))
        for channel, site in enumerate(array.sites):
            gain = site_gain(event.xyz_mm, site, cfg.attenuation_mm)
            scaled = event.amplitude * gain * kernel
            start = center - peak_index
            stop = start + kernel_n
            src_start = 0
            dst_start = start
            if dst_start < 0:
                src_start = -dst_start
                dst_start = 0
            src_stop = kernel_n
            dst_stop = stop
            if dst_stop > n_samples:
                src_stop -= dst_stop - n_samples
                dst_stop = n_samples
            if dst_stop <= dst_start or src_stop <= src_start:
                continue
            voltages[dst_start:dst_stop, channel] += scaled[src_start:src_stop]

    if cfg.noise_uv > 0:
        voltages += rng.normal(0.0, cfg.noise_uv, size=voltages.shape)
    if not np.isfinite(voltages).all():
        raise ValueError("recording model produced non-finite samples")
    return voltages, array
