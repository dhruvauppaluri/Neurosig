"""Toy stochastic neural activity. Not a biophysical network model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from viz.sim.events import NeuralEvent
from viz.sim.regions import REGIONS, get_region


@dataclass(frozen=True)
class PopulationConfig:
    population_id: str
    region_id: str
    n_units: int
    mean_rate_hz: float
    burstiness: float = 0.15
    correlation: float = 0.08
    amplitude_uv: float = -80.0
    duration_ms: float = 1.0
    amplitude_jitter: float = 0.25


DEFAULT_POPULATIONS: tuple[PopulationConfig, ...] = (
    PopulationConfig("frontal_l5", "frontal_left", 48, 4.0, 0.22, 0.12, -90.0),
    PopulationConfig("frontal_r5", "frontal_right", 48, 4.0, 0.22, 0.12, -90.0),
    PopulationConfig("motor_l", "motor_left", 32, 8.0, 0.35, 0.2, -110.0),
    PopulationConfig("motor_r", "motor_right", 32, 8.0, 0.35, 0.2, -110.0),
    PopulationConfig("parietal_l", "parietal_left", 36, 5.0, 0.18, 0.1, -75.0),
    PopulationConfig("parietal_r", "parietal_right", 36, 5.0, 0.18, 0.1, -75.0),
    PopulationConfig("temporal_l", "temporal_left", 28, 3.5, 0.12, 0.05, -70.0),
    PopulationConfig("temporal_r", "temporal_right", 28, 3.5, 0.12, 0.05, -70.0),
    PopulationConfig("vis_ctx", "visual_cortex", 40, 6.0, 0.28, 0.25, -95.0),
    PopulationConfig("hpc_l", "hippocampus_left", 20, 2.5, 0.45, 0.3, -120.0),
    PopulationConfig("hpc_r", "hippocampus_right", 20, 2.5, 0.45, 0.3, -120.0),
    PopulationConfig("thal", "thalamus", 24, 7.0, 0.1, 0.15, -60.0),
)


@dataclass
class SyntheticUnit:
    unit_id: str
    population_id: str
    region_id: str
    xyz_mm: tuple[float, float, float]
    base_rate_hz: float
    burstiness: float
    amplitude_uv: float
    duration_ms: float
    next_spike_s: float
    burst_remaining: int = 0


@dataclass
class SyntheticGenerator:
    seed: int
    rate_scale: float = 1.0
    populations: tuple[PopulationConfig, ...] = DEFAULT_POPULATIONS
    rng: np.random.Generator = field(init=False)
    units: list[SyntheticUnit] = field(init=False)
    _region_gain_until: dict[str, float] = field(init=False)
    _region_gain: dict[str, float] = field(init=False)
    spike_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.units = []
        self._region_gain_until = {region_id: 0.0 for region_id in REGIONS}
        self._region_gain = {region_id: 1.0 for region_id in REGIONS}
        self.spike_index = {}
        for population in self.populations:
            region = get_region(population.region_id)
            for index in range(population.n_units):
                unit_id = f"{population.population_id}:{index}"
                rate = max(0.05, float(self.rng.lognormal(np.log(population.mean_rate_hz), 0.35)))
                amplitude = float(
                    population.amplitude_uv
                    * (1.0 + population.amplitude_jitter * self.rng.uniform(-1.0, 1.0))
                )
                unit = SyntheticUnit(
                    unit_id=unit_id,
                    population_id=population.population_id,
                    region_id=population.region_id,
                    xyz_mm=region.sample(self.rng),
                    base_rate_hz=rate,
                    burstiness=population.burstiness,
                    amplitude_uv=amplitude,
                    duration_ms=population.duration_ms,
                    next_spike_s=self._draw_isi(rate, burst=False),
                    burst_remaining=0,
                )
                self.units.append(unit)
                self.spike_index[unit_id] = 0

        for population in self.populations:
            if population.correlation <= 0:
                continue
            region = get_region(population.region_id)
            members = [unit for unit in self.units if unit.population_id == population.population_id]
            shared = region.sample(self.rng)
            for unit in members:
                mix = 0.35 * population.correlation
                mixed = (1.0 - mix) * np.asarray(unit.xyz_mm) + mix * np.asarray(shared)
                unit.xyz_mm = (float(mixed[0]), float(mixed[1]), float(mixed[2]))

    def _draw_isi(self, rate_hz: float, burst: bool) -> float:
        rate = max(rate_hz, 0.05)
        if burst:
            rate = max(rate, 250.0)
        return float(self.rng.exponential(1.0 / rate))

    def _gain(self, region_id: str, t_s: float, correlation: float) -> float:
        if t_s >= self._region_gain_until[region_id]:
            self._region_gain_until[region_id] = t_s + 0.02
            if correlation <= 0:
                self._region_gain[region_id] = 1.0
            else:
                self._region_gain[region_id] = float(
                    self.rng.lognormal(mean=0.0, sigma=max(correlation, 1e-6))
                )
        return self._region_gain[region_id]

    def _population_correlation(self, population_id: str) -> float:
        for population in self.populations:
            if population.population_id == population_id:
                return population.correlation
        return 0.0

    def generate(self, t_start_s: float, t_end_s: float) -> list[NeuralEvent]:
        if t_end_s <= t_start_s:
            return []
        events: list[NeuralEvent] = []
        for unit in self.units:
            correlation = self._population_correlation(unit.population_id)
            while unit.next_spike_s < t_end_s:
                if unit.next_spike_s >= t_start_s:
                    gain = self._gain(unit.region_id, unit.next_spike_s, correlation)
                    if self.rng.random() > min(1.0, 0.35 + 0.65 / max(gain, 1e-6)):
                        unit.next_spike_s += self._draw_isi(unit.base_rate_hz * self.rate_scale, False)
                        continue
                    index = self.spike_index[unit.unit_id]
                    self.spike_index[unit.unit_id] = index + 1
                    events.append(
                        NeuralEvent(
                            event_id=f"syn:{unit.unit_id}:{index}",
                            t_s=float(unit.next_spike_s),
                            xyz_mm=unit.xyz_mm,
                            region_id=unit.region_id,
                            amplitude=unit.amplitude_uv,
                            source_id=unit.unit_id,
                            kind="synthetic_source",
                            duration_ms=unit.duration_ms,
                            coordinate_frame="stylized_human",
                            position_quality="simulated",
                            amplitude_units="uv",
                            metadata={
                                "model": "toy_stochastic",
                                "population_id": unit.population_id,
                                "not_physiology": True,
                            },
                        )
                    )
                    if unit.burst_remaining > 0:
                        unit.burst_remaining -= 1
                        unit.next_spike_s += self._draw_isi(unit.base_rate_hz * self.rate_scale, True)
                    else:
                        if self.rng.random() < unit.burstiness:
                            unit.burst_remaining = int(self.rng.integers(2, 6))
                            unit.next_spike_s += self._draw_isi(unit.base_rate_hz * self.rate_scale, True)
                        else:
                            unit.next_spike_s += self._draw_isi(unit.base_rate_hz * self.rate_scale, False)
                else:
                    unit.next_spike_s += self._draw_isi(unit.base_rate_hz * self.rate_scale, False)
        events.sort(key=lambda event: (event.t_s, event.event_id))
        return events
