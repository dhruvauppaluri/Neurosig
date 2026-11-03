"""Synthetic localization benchmark: noise and overlap sweeps.

Uses the quasi-static monopole in ``edge.forward``, not LFPy/MEArec biophysics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from edge import SAMPLE_RATE_HZ, SEED, SNIPPET_POST_MS, SNIPPET_PRE_MS
from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.localize import AmplitudeLocalizer, BeamformLocalizer, TemplateLocalizer
from edge.localize.template import default_training_xyz_um
from edge.snippets import extract_snippet
from edge.types import snippet_sample_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
MEA_REC_PATH = REPO_ROOT / "data" / "smoke" / "recording.h5"

# Held-out relative to default_training_xyz_um (xs ±25/15/35..., ys step 40, zs ±25/0).
HELD_OUT_XYZ_UM = (
    (30.0, 300.0, 0.0),
    (22.0, 170.0, 8.0),
    (48.0, 450.0, -12.0),
    (18.0, 90.0, 4.0),
)

NOISE_UV = (1.0, 8.0, 25.0, 60.0)
OVERLAP_OFFSET_SAMPLES = (None, 0, 10)
AMPLITUDE_UV = 12.0
N_REPEATS = 4


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mearec_section() -> dict[str, Any]:
    path = MEA_REC_PATH
    if not path.exists():
        return {
            "skipped": True,
            "path": str(path.relative_to(REPO_ROOT)),
            "limitation": (
                "MEArec smoke recording is absent. This benchmark uses the simplified "
                "quasi-static monopole forward model only and does not measure LFPy/"
                "MEArec biophysical localization accuracy."
            ),
        }
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
        return {
            "skipped": True,
            "path": str(path.relative_to(REPO_ROOT)),
            "h5_keys": keys,
            "limitation": (
                "MEArec recording is present but was not used for error metrics. "
                "The engines assume the quasi-static monopole on linear_probe_um, "
                "which is not LFPy/MEArec accuracy."
            ),
        }
    except Exception as exc:  # pragma: no cover - optional path
        return {
            "skipped": True,
            "path": str(path.relative_to(REPO_ROOT)),
            "limitation": f"MEArec file present but unreadable ({exc}). Forward-model sweep only.",
        }


def run_localization_benchmark(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    array = linear_probe_um()
    n_pre, n_post = snippet_sample_counts(SAMPLE_RATE_HZ)
    sample_rate = SAMPLE_RATE_HZ
    training = default_training_xyz_um(array.n_channels)
    engines = {
        "amplitude": AmplitudeLocalizer(),
        "template": TemplateLocalizer(array=array, training_xyz_um=training),
        "beamform": BeamformLocalizer(array=array),
    }

    errors: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    isolated_errors: dict[str, list[float]] = defaultdict(list)
    isolated_latencies: dict[str, list[float]] = defaultdict(list)
    by_condition: list[dict[str, Any]] = []
    n_available = 0
    n_estimates = 0

    pad = n_pre + n_post + 64
    for loc_index, xyz in enumerate(HELD_OUT_XYZ_UM):
        for noise_uv in NOISE_UV:
            for overlap in OVERLAP_OFFSET_SAMPLES:
                trial_errors: dict[str, list[float]] = defaultdict(list)
                trial_latencies: dict[str, list[float]] = defaultdict(list)
                for repeat in range(N_REPEATS):
                    t_peak = n_pre + 40
                    sources = [
                        PointSource(t_sample=t_peak, xyz_um=xyz, amplitude_uv=AMPLITUDE_UV),
                    ]
                    if overlap is not None:
                        other = (
                            xyz[0] + 20.0,
                            min(620.0, xyz[1] + 80.0),
                            xyz[2] - 6.0,
                        )
                        sources.append(
                            PointSource(
                                t_sample=t_peak + int(overlap),
                                xyz_um=other,
                                amplitude_uv=AMPLITUDE_UV,
                            )
                        )
                    n_samples = t_peak + n_post + pad
                    voltages = render_recording(
                        sources,
                        array,
                        sample_rate,
                        float(noise_uv),
                        seed=int(rng.integers(0, 2**31 - 1)),
                        n_samples=n_samples,
                    )
                    snippet = extract_snippet(
                        voltages,
                        t_peak,
                        array,
                        event_id=repeat,
                        sample_rate_hz=sample_rate,
                        n_pre=n_pre,
                        n_post=n_post,
                        ground_truth_xyz_um=xyz,
                    )
                    for name, engine in engines.items():
                        estimate = engine.estimate(snippet)
                        n_estimates += 1
                        if estimate.available:
                            n_available += 1
                        if estimate.latency_s is not None:
                            latencies[name].append(float(estimate.latency_s))
                            trial_latencies[name].append(float(estimate.latency_s))
                        if estimate.error_um is not None:
                            errors[name].append(float(estimate.error_um))
                            trial_errors[name].append(float(estimate.error_um))
                        if overlap is None and float(noise_uv) <= 1.0:
                            if estimate.error_um is not None:
                                isolated_errors[name].append(float(estimate.error_um))
                            if estimate.latency_s is not None:
                                isolated_latencies[name].append(float(estimate.latency_s))
                by_condition.append(
                    {
                        "location_um": list(xyz),
                        "location_index": loc_index,
                        "noise_uv": float(noise_uv),
                        "overlap_offset_samples": overlap,
                        "n_repeats": N_REPEATS,
                        "median_error_um": {name: _median(trial_errors[name]) for name in engines},
                        "median_latency_s": {name: _median(trial_latencies[name]) for name in engines},
                    }
                )

    methods = {}
    methods_low_noise_isolated = {}
    for name in engines:
        methods[name] = {
            "median_error_um": _median(errors[name]),
            "p90_error_um": float(np.percentile(errors[name], 90)) if errors[name] else None,
            "median_latency_s": _median(latencies[name]),
            "n": len(errors[name]),
        }
        methods_low_noise_isolated[name] = {
            "median_error_um": _median(isolated_errors[name]),
            "p90_error_um": float(np.percentile(isolated_errors[name], 90)) if isolated_errors[name] else None,
            "median_latency_s": _median(isolated_latencies[name]),
            "n": len(isolated_errors[name]),
            "selection": "noise_uv<=1 and no overlapping source",
        }

    return {
        "seed": seed,
        "sample_rate_hz": sample_rate,
        "n_channels": array.n_channels,
        "snippet_pre_ms": SNIPPET_PRE_MS,
        "snippet_post_ms": SNIPPET_POST_MS,
        "lambda_um": 80.0,
        "forward_model": "simplified quasi-static monopole (not LFPy/MEArec)",
        "held_out_locations_um": [list(xyz) for xyz in HELD_OUT_XYZ_UM],
        "training_template_count": int(training.shape[0]),
        "noise_uv": list(NOISE_UV),
        "overlap_offset_samples": list(OVERLAP_OFFSET_SAMPLES),
        "amplitude_uv": AMPLITUDE_UV,
        "n_repeats": N_REPEATS,
        "n_estimates": n_estimates,
        "n_available": n_available,
        "methods": methods,
        "methods_low_noise_isolated": methods_low_noise_isolated,
        "conditions": by_condition,
        "mearec": _mearec_section(),
        "units": {
            "error": "micrometers",
            "latency": "seconds",
            "noise": "microvolts",
        },
    }
