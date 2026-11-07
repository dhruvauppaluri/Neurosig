"""Phase 8 synthetic event generator with continuous source positions and disjoint seed pools.

Pools: dev (tuning), select (model selection), final (touched once). Seeds never overlap.
Primary population: single source (``interferer=False``) plus a ``contaminated`` subset with
a second, weaker-or-equal source (distance 30..90 um along y, 3..14 samples later) that is
scored against the first source only, mirroring the v3 benchmark's difficult cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge.benchmark_v3 import _array
from edge.forward import PointSource, render_recording
from edge.localize.template import default_training_xyz_um
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, Snippet

POOL_SEEDS = {"dev": 20268001, "select": 20268002, "final": 20268003}
NOISE_LEVELS = (2.0, 10.0, 25.0)


@dataclass(frozen=True)
class SynthEvent:
    event_id: int
    pool: str
    geometry: str
    noise_uv: float
    contaminated: bool
    truth_xyz_um: tuple[float, float, float]
    sources_xyz_um: tuple[tuple[float, float, float], ...]
    snippet: Snippet
    render_model: str = "offset_reciprocal"


def _render(sources, array, noise, seed, model):
    if model == "offset_reciprocal":
        return render_recording(sources, array, 32000.0, noise, seed, 160)
    if model == "line_source":
        # Same kernel, line-source gain. Implemented by pre-scaling amplitudes per channel.
        from edge.forward import predicted_amplitude_patterns, triphasic_template
        electrodes = array.xyz_um()
        kernel = triphasic_template(80)
        peak = int(np.argmin(kernel))
        v = np.zeros((160, array.n_channels))
        for s in sources:
            g = predicted_amplitude_patterns(np.asarray(s.xyz_um)[None], electrodes, model="line_source")[0]
            g = g * 813.0  # matches the offset_reciprocal gain at r=20 um (10.0 vs 1/sqrt(20^2+80^2))
            start = s.t_sample - peak
            v[start:start + 80] += (s.amplitude_uv * g[:, None] * kernel[None, :]).T
        if noise > 0:
            v += np.random.default_rng(seed).normal(0.0, noise, v.shape)
        return v
    raise ValueError(model)


def make_events(pool: str, n_per_cell: int = 12, geometry: str = "linear", render_model: str = "offset_reciprocal",
                contaminated_fraction: float = 0.5, jitter_samples: int = 1) -> list[SynthEvent]:
    """n_per_cell events per noise level. Positions are continuous and never within 1 um of a catalog site."""
    array: ElectrodeArray = _array(geometry)
    rng = np.random.default_rng(POOL_SEEDS[pool] + (0 if geometry == "linear" else 7))
    catalog = default_training_xyz_um(array.n_channels)
    events: list[SynthEvent] = []
    for noise in NOISE_LEVELS:
        for _ in range(n_per_cell):
            while True:
                truth = (float(rng.uniform(12, 80)), float(rng.uniform(80, 560)), float(rng.uniform(-30, 30)))
                if np.min(np.linalg.norm(catalog - np.asarray(truth), axis=1)) >= 1.0:
                    break
            amp = float(rng.uniform(9.0, 16.0))
            jit = int(rng.integers(-jitter_samples, jitter_samples + 1))
            sources = [PointSource(64 + jit, truth, amp)]
            positions = [truth]
            contaminated = bool(rng.random() < contaminated_fraction)
            if contaminated:
                dy = float(rng.choice([-1, 1]) * rng.uniform(30, 90))
                other = (float(rng.uniform(12, 80)), float(np.clip(truth[1] + dy, 20, 620)), float(rng.uniform(-30, 30)))
                dt = int(rng.choice([-1, 1]) * rng.integers(3, 15))
                sources.append(PointSource(64 + jit + dt, other, float(amp * rng.uniform(0.5, 1.0))))
                positions.append(other)
            trace = _render(sources, array, noise, int(rng.integers(0, 2**31)), render_model)
            eid = len(events)
            snippet = extract_snippet(trace, 64, array, event_id=eid, ground_truth_xyz_um=truth)
            events.append(SynthEvent(eid, pool, geometry, noise, contaminated, truth, tuple(positions), snippet,
                                     render_model))
    return events
