"""Multi-seed held-out localization study plus a small MEArec biophysical check."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import h5py
import numpy as np

from edge.forward import PointSource, render_recording
from edge.geometry import linear_probe_um
from edge.localize import AmplitudeLocalizer, BeamformLocalizer, TemplateLocalizer
from edge.localize.template import default_training_xyz_um
from edge.localize.weighted import WeightedScanLocalizer
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, ElectrodeSite
from scripts.detect_spikes import load_ground_truth


def _array(geometry: str) -> ElectrodeArray:
    if geometry == "linear":
        return linear_probe_um()
    return ElectrodeArray(tuple(ElectrodeSite(i, (-10.0 if i % 2 else 10.0,
                                                i * 20.0, 0.0)) for i in range(32)),
                          "staggered two-column synthetic array")


def _engines(array: ElectrodeArray) -> dict:
    training = default_training_xyz_um(array.n_channels)
    return {
        "amplitude": AmplitudeLocalizer(),
        "template": TemplateLocalizer(array=array, training_xyz_um=training),
        "beamform": BeamformLocalizer(array=array),
        "weighted_scan": WeightedScanLocalizer(array=array),
    }


def _summary(rows: list[dict]) -> dict:
    out = {}
    for method in ("amplitude", "template", "beamform", "weighted_scan"):
        selected = [r for r in rows if r["method"] == method]
        errors = np.asarray([r["error_um"] for r in selected if r["error_um"] is not None])
        out[method] = {
            "n": len(selected), "failures": len(selected) - len(errors),
            "median_error_um": float(np.median(errors)) if len(errors) else None,
            "p90_error_um": float(np.percentile(errors, 90)) if len(errors) else None,
            "median_runtime_ms": float(np.median([r["runtime_ms"] for r in selected])) if selected else None,
        }
    return out


def run_synthetic(seeds: tuple[int, ...] = (20261103, 20261104, 20261105)) -> dict:
    rows: list[dict] = []
    for geometry in ("linear", "staggered"):
        array = _array(geometry)
        engines = _engines(array)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for distance in (20.0, 45.0, 75.0):
                for y in (110.0, 310.0, 510.0):
                    # Positions deliberately exclude the training grid.
                    truth = (distance + 2.0, y + 7.0, 8.0)
                    for noise in (2.0, 10.0, 25.0):
                        for overlap in (None, 8):
                            sources = [PointSource(64, truth, 12.0)]
                            if overlap is not None:
                                sources.append(PointSource(64 + overlap,
                                                           (distance + 15.0, y + 55.0, -10.0), 12.0))
                            trace = render_recording(sources, array, 32000.0, noise,
                                                     int(rng.integers(0, 2**31)), 160)
                            snippet = extract_snippet(trace, 64, array, event_id=len(rows),
                                                      ground_truth_xyz_um=truth)
                            for name, engine in engines.items():
                                start = time.perf_counter()
                                estimate = engine.estimate(snippet)
                                rows.append({"geometry": geometry, "seed": seed,
                                             "distance_um": distance, "noise_uv": noise,
                                             "overlap_samples": overlap, "method": name,
                                             "error_um": estimate.error_um if estimate.available else None,
                                             "runtime_ms": (time.perf_counter() - start) * 1000})
    conditions = {}
    for key in ("geometry", "distance_um", "noise_uv", "overlap_samples"):
        conditions[key] = {str(value): _summary([r for r in rows if r[key] == value])
                           for value in sorted({r[key] for r in rows}, key=str)}
    return {"seeds": list(seeds), "n_trials": len(rows) // 4,
            "training_test_policy": "fixed template grid; all test xyz offset from grid; identical voltage for every method",
            "forward_model": "quasi-static monopole; matched scans use this same model",
            "methods": _summary(rows), "conditions": conditions,
            "complexity": {"amplitude": "O(C)", "template": "O(C*T)",
                           "beamform": "O(C*G)", "weighted_scan": "O(C*G) per event; weights recomputed"},
            "timing_note": "wall-clock Python runtime includes implementation overhead, not ASIC latency"}


def run_mearec(path: Path = Path("data/smoke/recording.h5")) -> dict:
    if not path.exists():
        return {"status": "skipped", "reason": f"missing {path}"}
    with h5py.File(path) as handle:
        trace = np.asarray(handle["recordings"], dtype=float)
        positions = np.asarray(handle["channel_positions"], dtype=float)
        locations = np.asarray(handle["template_locations"], dtype=float)
    if locations.shape != (1, 3):
        return {"status": "skipped", "reason": "smoke check requires one known template source"}
    # Rotate the probe's original z shank into the localizer's y axis; preserve
    # all relative distances and do not mix these micrometers with display mm.
    origin_z = float(np.min(positions[:, 2]))
    transform = lambda p: (float(p[0]), float(p[2] - origin_z), float(p[1]))
    array = ElectrodeArray(tuple(ElectrodeSite(i, transform(p)) for i, p in enumerate(positions)),
                           "MEArec Neuronexus-32 original positions rotated into probe_um")
    truth = transform(locations[0])
    engines = _engines(array)
    rows = []
    for i, t in enumerate(load_ground_truth(path)):
        snippet = extract_snippet(trace, int(round(t * 32000)), array, event_id=i,
                                  ground_truth_xyz_um=truth)
        for name, engine in engines.items():
            estimate = engine.estimate(snippet)
            rows.append({"method": name, "error_um": estimate.error_um if estimate.available else None,
                         "runtime_ms": estimate.latency_s * 1000})
    return {"status": "ran", "dataset": str(path), "n_spikes": len(rows) // 4,
            "known_simulated_source_xyz_um": list(truth), "methods": _summary(rows),
            "limitation": "one simulated neuron and nine spikes; templates/scans assume monopole despite MEArec biophysical waveform"}


def main() -> None:
    report = {"synthetic": run_synthetic(), "mearec": run_mearec()}
    target = Path("reports/localization-benchmark-v2.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"synthetic": report["synthetic"]["methods"],
                      "mearec": report["mearec"].get("methods")}, indent=2))


if __name__ == "__main__":
    main()
