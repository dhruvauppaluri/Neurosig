"""Session engine: clock, modes, metrics, and event generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Literal

from viz.sim.events import EventBatch, NeuralEvent
from viz.sim.golden_bridge import process_window, refuse_spike_timestamps_as_voltage
from viz.sim.localization import ElectrodeArray, default_probe_array, empty_session_localization
from viz.sim.neuropixels import DISCLAIMER, NeuropixelsPlayer, NeuropixelsRecording, load_recording
from viz.sim.regions import region_catalog
from viz.sim.synthetic import DEFAULT_POPULATIONS, SyntheticGenerator

Mode = Literal["synthetic", "neuropixels", "hybrid"]


@dataclass
class SessionMetrics:
    simulated_events: int = 0
    recorded_spikes: int = 0
    detections: int = 0
    missed: int = 0
    precision: float | None = None
    recall: float | None = None
    active_channels: int = 32
    localizations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulated_event_count": self.simulated_events,
            "recorded_spike_count": self.recorded_spikes,
            "detection_count": self.detections,
            "missed_count": self.missed,
            "precision": self.precision,
            "recall": self.recall,
            "active_channels": self.active_channels,
            "localization_count": self.localizations,
        }


@dataclass
class SessionState:
    mode: Mode = "synthetic"
    seed: int = 42
    running: bool = True
    speed: float = 1.0
    rate_scale: float = 1.0
    t_s: float = 0.0
    visible_kinds: tuple[str, ...] = (
        "synthetic_source",
        "neuropixels_spike",
        "electrode_activity",
        "detected_spike",
        "missed_spike",
        "localization_estimate",
    )
    selected_region: str | None = None
    selected_channel: int | None = None
    run_golden_model: bool = True
    detector_note: str = "Golden-model detections are offline windowed software results, not RTL."


class NeuralEventEngine:
    def __init__(self, recording: NeuropixelsRecording | None = None) -> None:
        self.lock = threading.Lock()
        self.state = SessionState()
        self.metrics = SessionMetrics()
        self.synthetic = SyntheticGenerator(seed=self.state.seed)
        self.recording = recording or load_recording()
        self.player = NeuropixelsPlayer(self.recording)
        self.player.reset()
        self.electrodes: ElectrodeArray = default_probe_array()
        self.timeline: list[NeuralEvent] = []
        self.snippets: dict[str, dict[str, Any]] = {}
        self.localization: dict[str, Any] = empty_session_localization()
        self._last_wall = time.monotonic()
        self._detector_cursor = 0.0
        self._localization_honesty = (
            "Simplified 1/r amplitude localization in the stylized viewer; "
            "not biophysics and not human neuron positions. Estimates stay unverified until a report exists."
        )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema_version": 1,
                "mode": self.state.mode,
                "seed": self.state.seed,
                "running": self.state.running,
                "speed": self.state.speed,
                "rate_scale": self.state.rate_scale,
                "t_s": self.state.t_s,
                "visible_kinds": list(self.state.visible_kinds),
                "selected_region": self.state.selected_region,
                "selected_channel": self.state.selected_channel,
                "run_golden_model": self.state.run_golden_model,
                "detector_note": self.state.detector_note,
                "metrics": self.metrics.to_dict(),
                "regions": region_catalog(),
                "electrodes": self.electrodes.to_dict(),
                "neuropixels": self.recording.to_summary(),
                "disclaimer": DISCLAIMER,
                "localization": dict(self.localization),
                "populations": [
                    {
                        "population_id": pop.population_id,
                        "region_id": pop.region_id,
                        "n_units": pop.n_units,
                        "mean_rate_hz": pop.mean_rate_hz,
                    }
                    for pop in DEFAULT_POPULATIONS
                ],
                "honesty": {
                    "synthetic": "Toy stochastic populations with region-dependent rates, bursts, and weak correlation.",
                    "neuropixels": DISCLAIMER,
                    "golden_model": self.state.detector_note,
                    "localization": self._localization_honesty,
                },
            }

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "mode" in payload:
                mode = payload["mode"]
                if mode not in {"synthetic", "neuropixels", "hybrid"}:
                    raise ValueError(f"unknown mode {mode}")
                if mode != self.state.mode:
                    self.metrics = SessionMetrics()
                    self.timeline.clear()
                self.state.mode = mode
            if "running" in payload:
                self.state.running = bool(payload["running"])
                self._last_wall = time.monotonic()
            if payload.get("reset"):
                self._reset_locked(int(payload.get("seed", self.state.seed)))
            if "seed" in payload and not payload.get("reset"):
                self._reset_locked(int(payload["seed"]))
            if "speed" in payload:
                self.state.speed = max(0.05, min(float(payload["speed"]), 20.0))
            if "rate_scale" in payload:
                self.state.rate_scale = max(0.05, min(float(payload["rate_scale"]), 8.0))
                self.synthetic.rate_scale = self.state.rate_scale
            if "visible_kinds" in payload:
                self.state.visible_kinds = tuple(payload["visible_kinds"])
            if "selected_region" in payload:
                self.state.selected_region = payload["selected_region"]
            if "selected_channel" in payload:
                self.state.selected_channel = payload["selected_channel"]
            if "run_golden_model" in payload:
                self.state.run_golden_model = bool(payload["run_golden_model"])
            if "seek_s" in payload:
                target = max(0.0, float(payload["seek_s"]))
                self._reset_locked(self.state.seed)
                if target > 0:
                    detect = self.state.run_golden_model
                    self.state.run_golden_model = False
                    self._generate_range(0.0, target)
                    self.state.run_golden_model = detect
                self.timeline.clear()
                self.state.t_s = target
                self._detector_cursor = target
        return self.snapshot()

    def _reset_locked(self, seed: int) -> None:
        self.state.seed = seed
        self.state.t_s = 0.0
        self.state.running = True
        self.metrics = SessionMetrics()
        self.synthetic.rate_scale = self.state.rate_scale
        self.synthetic.reset(seed)
        self.player.reset()
        self.timeline.clear()
        self.snippets.clear()
        self.localization = empty_session_localization()
        self._last_wall = time.monotonic()
        self._detector_cursor = 0.0

    def step_wall(self, dt_wall_s: float | None = None) -> EventBatch:
        with self.lock:
            now = time.monotonic()
            if dt_wall_s is None:
                dt_wall_s = now - self._last_wall
            self._last_wall = now
            if not self.state.running:
                return EventBatch(self.state.t_s, self.state.t_s, [])
            dt = max(0.0, min(dt_wall_s * self.state.speed, 0.25))
            t0 = self.state.t_s
            t1 = t0 + dt
            events = self._generate_range(t0, t1)
            self.state.t_s = t1
            self._update_metrics(events)
            self.timeline.extend(events)
            if len(self.timeline) > 20_000:
                self.timeline = self.timeline[-10_000:]
            return EventBatch(t0, t1, events)

    def generate_range(self, t0: float, t1: float) -> list[NeuralEvent]:
        with self.lock:
            return self._generate_range(t0, t1)

    def _generate_range(self, t0: float, t1: float) -> list[NeuralEvent]:
        events: list[NeuralEvent] = []
        if self.state.mode in {"synthetic", "hybrid"}:
            events.extend(self.synthetic.generate(t0, t1))
        if self.state.mode in {"neuropixels", "hybrid"}:
            events.extend(self.player.generate(t0, t1))
        if self.state.run_golden_model and self.state.mode in {"synthetic", "hybrid"}:
            sources = [
                event
                for event in events
                if event.kind == "synthetic_source" and event.region_id == "visual_cortex"
            ]
            if sources and (t1 - self._detector_cursor) >= 0.05:
                window_start = self._detector_cursor
                window_dur = max(t1 - window_start, 0.05)
                try:
                    result = process_window(sources, window_start, window_dur)
                    events.extend(result.events)
                    self.localization = dict(result.localization)
                    self.snippets.update(result.snippets)
                    if result.metrics.get("ground_truth_available"):
                        self.metrics.precision = float(result.metrics.get("precision") or 0.0)
                        self.metrics.recall = float(result.metrics.get("recall") or 0.0)
                except ValueError:
                    pass
                self._detector_cursor = t1
        events.sort(key=lambda event: (event.t_s, event.event_id))
        if self.state.selected_region:
            events = [event for event in events if event.region_id == self.state.selected_region]
        if self.state.selected_channel is not None:
            events = [
                event
                for event in events
                if event.channel_id is None or event.channel_id == self.state.selected_channel
            ]
        visible = set(self.state.visible_kinds)
        return [event for event in events if event.kind in visible]

    def _update_metrics(self, events: list[NeuralEvent]) -> None:
        for event in events:
            if event.kind == "synthetic_source":
                self.metrics.simulated_events += 1
            elif event.kind == "neuropixels_spike":
                self.metrics.recorded_spikes += 1
            elif event.kind == "detected_spike":
                self.metrics.detections += 1
            elif event.kind == "missed_spike":
                self.metrics.missed += 1
            elif event.kind == "localization_estimate":
                self.metrics.localizations += 1

    def get_snippet(self, event_id: str) -> dict[str, Any] | None:
        with self.lock:
            payload = self.snippets.get(event_id)
            return None if payload is None else dict(payload)

    def timeline_points(self, limit: int = 400) -> list[dict[str, Any]]:
        with self.lock:
            sample = self.timeline[-limit:]
            return [
                {"t_s": event.t_s, "kind": event.kind, "region_id": event.region_id}
                for event in sample
            ]


def assert_not_adc_from_spikes(events: list[NeuralEvent]) -> None:
    for event in events:
        if event.kind == "neuropixels_spike":
            refuse_spike_timestamps_as_voltage(event.kind)
