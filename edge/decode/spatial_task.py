"""Synthetic 2-class spatial cluster decoding task (seed 20261103).

Labels choose which of two source clusters along the probe y-axis is more
active. Ground-truth source locations and spike times are generated first,
then rendered with ``edge.forward.render_recording`` (quasi-static monopole
stub, not LFPy/MEArec).

Localization features prefer an importable ``LocalizationEngine``. If that
path is unavailable, features use the observed peak-channel contact position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

for _thread_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import numpy as np

from edge import N_CHANNELS, SAMPLE_RATE_HZ, SEED
from edge.decode.features import SPATIAL_DIM, stack_baseline, stack_spatial
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.localize.base import LocalizationEngine, NotImplementedLocalizer
from edge.snippets import extract_snippet
from edge.types import LocalizationEstimate, Snippet

WINDOW_S = 0.1
N_EXAMPLES = 4096
N_TRAIN = 3072
N_TEST = 1024
HIGH_RATE_HZ = 16.0
LOW_RATE_HZ = 8.0
TONIC_RATE_HZ = 10.0
JITTER_SIGMA_UM = 25.0
NOISE_UV = 3.0
AMPLITUDE_UV = 10.0

FORWARD_MODEL = (
    "edge.forward.render_recording: quasi-static point-monopole voltages with a "
    "triphasic kernel (not LFPy/MEArec morphology)"
)
LOCALIZER_FALLBACK = "observed_peak_channel_electrode_position"


@dataclass(frozen=True)
class Source:
    source_id: int
    cluster: int
    xyz_um: tuple[float, float, float]


@dataclass
class SpatialDataset:
    seed: int
    sources: tuple[Source, ...]
    labels: np.ndarray
    counts: np.ndarray
    estimated_xyz: list[np.ndarray]
    true_xyz: list[np.ndarray]
    spike_times_s: list[np.ndarray]
    spike_channels: list[np.ndarray]
    used_stub_forward: bool
    localizer_method: str
    localizer_fallback: str


def make_sources(rng: np.random.Generator) -> tuple[Source, ...]:
    """Two clusters near different probe y positions, with small x/z scatter."""
    # Clusters sit on neighboring y ranges so channel counts overlap; tonic
    # sources fire independently of the label and raise the count baseline.
    deep_y = np.array([180.0, 200.0, 220.0, 240.0])
    shallow_y = np.array([300.0, 320.0, 340.0, 360.0])
    tonic_y = np.linspace(40.0, 580.0, 8)
    sources: list[Source] = []
    source_id = 0
    for cluster, ys in ((0, deep_y), (1, shallow_y), (-1, tonic_y)):
        for y in ys:
            xy_jitter = rng.uniform(-12.0, 12.0, size=2)
            sources.append(
                Source(
                    source_id=source_id,
                    cluster=cluster,
                    xyz_um=(float(xy_jitter[0]), float(y), float(xy_jitter[1])),
                )
            )
            source_id += 1
    return tuple(sources)


def _optional_localizer() -> LocalizationEngine | None:
    try:
        from edge.localize.amplitude import AmplitudeLocalizer
    except ImportError:
        return None
    engine = AmplitudeLocalizer()
    if isinstance(engine, NotImplementedLocalizer):
        return None
    return engine


def generate_dataset(
    seed: int = SEED,
    examples: int = N_EXAMPLES,
    jitter_sigma_um: float = JITTER_SIGMA_UM,
    localizer: LocalizationEngine | None = None,
) -> SpatialDataset:
    rng = np.random.default_rng(seed)
    del jitter_sigma_um  # legacy API parameter; no ground-truth jitter in benchmarks
    sources = make_sources(rng)
    array = linear_probe_um(n_channels=N_CHANNELS)
    n_samples = int(round(WINDOW_S * SAMPLE_RATE_HZ))
    engine = localizer if localizer is not None else _optional_localizer()
    method = engine.method if engine is not None else LOCALIZER_FALLBACK
    labels = rng.integers(0, 2, examples, dtype=np.int8)
    counts = np.zeros((examples, N_CHANNELS), dtype=np.uint8)
    estimated_xyz: list[np.ndarray] = []
    true_xyz: list[np.ndarray] = []
    spike_times_s: list[np.ndarray] = []
    spike_channels: list[np.ndarray] = []

    for i, label in enumerate(labels):
        times: list[float] = []
        xyz_true: list[tuple[float, float, float]] = []
        for source in sources:
            if source.cluster < 0:
                rate = TONIC_RATE_HZ
            elif source.cluster == int(label):
                rate = HIGH_RATE_HZ
            else:
                rate = LOW_RATE_HZ
            n_spikes = int(rng.poisson(rate * WINDOW_S))
            if n_spikes == 0:
                continue
            for t in rng.uniform(0.0, WINDOW_S, n_spikes):
                times.append(float(t))
                xyz_true.append(source.xyz_um)
        true_arr = np.asarray(xyz_true, dtype=np.float64).reshape(-1, 3)
        point_sources = [
            PointSource(
                t_sample=int(round(t * SAMPLE_RATE_HZ)),
                xyz_um=xyz,
                amplitude_uv=AMPLITUDE_UV,
            )
            for t, xyz in zip(times, xyz_true)
        ]
        voltages = render_recording(
            point_sources,
            array,
            SAMPLE_RATE_HZ,
            noise_uv=NOISE_UV,
            seed=int(seed) + 17 * (i + 1),
            n_samples=n_samples,
        )
        channels: list[int] = []
        est_points: list[tuple[float, float, float]] = []
        for event_id, (t, xyz, src) in enumerate(zip(times, xyz_true, point_sources)):
            snippet = extract_snippet(
                voltages,
                src.t_sample,
                array,
                event_id=event_id,
                ground_truth_xyz_um=xyz,
            )
            peak = snippet.peak_channel if snippet.peak_channel is not None else 0
            channels.append(int(peak))
            if engine is not None:
                estimate = engine.estimate(snippet)
                if estimate.available and estimate.xyz_um is not None:
                    est_points.append(estimate.xyz_um)
                    continue
            # Missing estimates fall back only to measured channel geometry.
            # Ground-truth source coordinates never enter model features.
            site = array.sites[int(peak)]
            est_points.append(site.xyz_um)
        if channels:
            ch_idx, ch_counts = np.unique(np.asarray(channels, dtype=np.int64), return_counts=True)
            row = np.zeros(N_CHANNELS, dtype=np.int64)
            row[ch_idx] = ch_counts
            counts[i] = np.clip(row, 0, 255).astype(np.uint8)
            est_arr = np.asarray(est_points, dtype=np.float64).reshape(-1, 3)
        else:
            est_arr = np.zeros((0, 3), dtype=np.float64)
        estimated_xyz.append(est_arr)
        true_xyz.append(true_arr)
        spike_times_s.append(np.asarray(times, dtype=np.float64))
        spike_channels.append(np.asarray(channels, dtype=np.int64))

    return SpatialDataset(
        seed=seed,
        sources=sources,
        labels=labels,
        counts=counts,
        estimated_xyz=estimated_xyz,
        true_xyz=true_xyz,
        spike_times_s=spike_times_s,
        spike_channels=spike_channels,
        used_stub_forward=True,
        localizer_method=method,
        localizer_fallback=LOCALIZER_FALLBACK,
    )


def feature_matrices(dataset: SpatialDataset) -> tuple[np.ndarray, np.ndarray]:
    baseline = stack_baseline(dataset.counts)
    spatial = stack_spatial(dataset.counts, dataset.estimated_xyz)
    if spatial.shape[1] != SPATIAL_DIM:
        raise ValueError("unexpected spatial feature width")
    return baseline, spatial


def split_indices(n: int = N_EXAMPLES, n_train: int = N_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    if n_train >= n:
        raise ValueError("n_train must be smaller than n")
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:]


def snippets_for_trial(dataset: SpatialDataset, trial: int, t0_sample: int = 0) -> list[Snippet]:
    """Rebuild snippets for one trial via the same forward model (for pipeline tests)."""
    array = linear_probe_um()
    times = dataset.spike_times_s[trial]
    xyz = dataset.true_xyz[trial]
    n_samples = int(round(WINDOW_S * SAMPLE_RATE_HZ))
    point_sources = [
        PointSource(t_sample=t0_sample + int(round(float(t) * SAMPLE_RATE_HZ)), xyz_um=tuple(loc), amplitude_uv=AMPLITUDE_UV)
        for t, loc in zip(times, xyz)
    ]
    voltages = render_recording(
        point_sources,
        array,
        SAMPLE_RATE_HZ,
        noise_uv=NOISE_UV,
        seed=int(dataset.seed) + 17 * (trial + 1),
        n_samples=max(n_samples, 1),
    )
    out: list[Snippet] = []
    for event_id, src in enumerate(point_sources):
        loc = tuple(float(v) for v in xyz[event_id])
        out.append(
            extract_snippet(
                voltages,
                src.t_sample,
                array,
                event_id=event_id,
                ground_truth_xyz_um=loc,
            )
        )
    return out


def jitter_estimates_from_ground_truth(
    snippets: list[Snippet],
    rng: np.random.Generator,
    sigma_um: float = JITTER_SIGMA_UM,
) -> list[LocalizationEstimate]:
    estimates: list[LocalizationEstimate] = []
    for snippet in snippets:
        gt = snippet.ground_truth_xyz_um
        if gt is None:
            estimates.append(
                LocalizationEstimate(
                    available=False,
                    method=LOCALIZER_FALLBACK,
                    xyz_um=None,
                    ground_truth_xyz_um=None,
                    error_um=None,
                    latency_s=0.0,
                    confidence=None,
                    reason="No ground-truth xyz on snippet; jitter fallback skipped",
                    event_id=snippet.event_id,
                    unverified=True,
                )
            )
            continue
        noise = rng.normal(0.0, sigma_um, size=3)
        xyz = (float(gt[0] + noise[0]), float(gt[1] + noise[1]), float(gt[2] + noise[2]))
        error = float(np.linalg.norm(np.asarray(xyz) - np.asarray(gt)))
        estimates.append(
            LocalizationEstimate(
                available=True,
                method=LOCALIZER_FALLBACK,
                xyz_um=xyz,
                ground_truth_xyz_um=gt,
                error_um=error,
                latency_s=0.0,
                confidence=float(np.exp(-error / max(sigma_um, 1.0))),
                reason="Simulated estimate: ground-truth xyz plus Gaussian jitter; not a validated localizer",
                event_id=snippet.event_id,
                unverified=True,
            )
        )
    return estimates


def source_table(sources: tuple[Source, ...]) -> list[dict[str, object]]:
    return [
        {
            "source_id": source.source_id,
            "cluster": source.cluster,
            "xyz_um": list(source.xyz_um),
        }
        for source in sources
    ]
