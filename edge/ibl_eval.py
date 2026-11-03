"""Multi-window IBL detection and real-recording localization evaluation."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from edge.ibl import DEFAULT_PID, SORTING_REVISION, load_raw_chunk
from edge.localize import LOCALIZATION_METHODS, get_localizer
from edge.localize.beamform import default_scan_grid_um
from edge.localize.template import default_training_xyz_um
from edge.snippets import extract_snippet
from scripts.detect_spikes import DetectorConfig, detect_candidates, estimate_noise, match_events, merge_channel_events
from scripts.filter_iir import IirConfig, design_sos
from scripts.real_detector import RealDetectorConfig, detect_real, evaluate_against_reference, sorter_aligned_snr
from scipy import signal


def _legacy(voltages_uv: np.ndarray, fs: float):
    filtered = signal.sosfilt(design_sos(IirConfig(sample_rate_hz=fs)), voltages_uv, axis=0)
    cfg = DetectorConfig(sample_rate_hz=fs)
    baseline, noise = estimate_noise(filtered)
    events = merge_channel_events(detect_candidates(filtered, baseline, noise, cfg), cfg)
    return events, cfg, filtered, baseline, noise


def _engines(array):
    y_shift = float(np.min(array.xyz_um()[:, 1]))
    training = default_training_xyz_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    scan = default_scan_grid_um(array.n_channels) + np.array([0.0, y_shift, 0.0])
    return {
        "amplitude": get_localizer("amplitude"),
        "template": get_localizer("template", array=array, training_xyz_um=training),
        "beamform": get_localizer("beamform", array=array, grid_xyz_um=scan),
        "weighted_scan": get_localizer("weighted_scan", array=array, grid_xyz_um=scan),
        "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan,
                                    model="line_source"),
    }


def evaluate_window(t0_s: float, duration_s: float = 1.0, pid: str = DEFAULT_PID,
                    max_sorter_spikes: int = 80) -> dict:
    chunk = load_raw_chunk(pid, t0_s, duration_s)
    fs = chunk["fs"]
    voltages = chunk["voltages_uv"]
    mask = chunk["sorter_mask"]
    truth = chunk["spike_times"][mask] - t0_s
    legacy_events, legacy_cfg, legacy_filtered, legacy_base, legacy_sigma = _legacy(voltages, fs)
    _, legacy_metrics = match_events(legacy_events, truth, legacy_cfg)
    variants = {
        "legacy_6sigma_no_car": RealDetectorConfig(
            sample_rate_hz=fs, use_car=False, threshold_sigma=6.0,
            artifact_channel_fraction=1.0, mad_iterations=0,
        ),
        "car_6sigma": RealDetectorConfig(sample_rate_hz=fs, use_car=True, threshold_sigma=6.0),
        "car_4sigma": RealDetectorConfig(sample_rate_hz=fs, use_car=True, threshold_sigma=4.0),
        "car_4sigma_hp_only": RealDetectorConfig(
            sample_rate_hz=fs, use_car=True, threshold_sigma=4.0, lowpass_hz=None,
        ),
        "car_4sigma_9000lp": RealDetectorConfig(
            sample_rate_hz=fs, use_car=True, threshold_sigma=4.0, lowpass_hz=9_000.0,
        ),
        "car_4sigma_both_polarity": RealDetectorConfig(
            sample_rate_hz=fs, use_car=True, threshold_sigma=4.0, polarity="both",
        ),
    }
    variant_reports = {}
    chosen = None
    for name, cfg in variants.items():
        result = detect_real(voltages, cfg)
        variant_reports[name] = {
            "events": result["merged_detections"],
            "candidates": result["channel_candidates"],
            "artifact_rejected": result["artifact_rejected_candidates"],
            "noise_sigma_uv": result["noise_sigma_uv"],
            "vs_sorting": evaluate_against_reference(result["events"], truth, cfg),
        }
        if name == "car_4sigma":
            chosen = result
    assert chosen is not None
    snr_legacy = sorter_aligned_snr(legacy_filtered, legacy_base, legacy_sigma, truth, fs)
    snr_improved = sorter_aligned_snr(chosen["filtered_uv"], chosen["baseline"], chosen["noise_sigma"], truth, fs)
    engines = _engines(chunk["array"])
    sorter_idx = np.flatnonzero(mask)[:max_sorter_spikes]
    loc_rows = []
    unit_estimates: dict[str, dict[int, list[tuple[float, float, float]]]] = {
        name: defaultdict(list) for name in engines
    }
    method_pairs: dict[str, list[float]] = defaultdict(list)
    for i, index in enumerate(sorter_idx):
        sample = int(round((chunk["spike_times"][index] - t0_s) * fs))
        peak_channel = int(chunk["cluster_channels"][chunk["spike_clusters"][index]])
        cluster = int(chunk["spike_clusters"][index])
        snippet = extract_snippet(
            voltages, sample, chunk["array"], event_id=i, sample_rate_hz=fs, peak_channel=peak_channel
        )
        ref_axial = float(chunk["locations"][peak_channel, 1])
        xyzs = {}
        for name, engine in engines.items():
            estimate = engine.estimate(snippet)
            xyz = estimate.xyz_um if estimate.available else None
            xyzs[name] = xyz
            loc_rows.append({
                "method": name,
                "available": bool(estimate.available and xyz is not None),
                "axial_disagreement_um": None if xyz is None else abs(float(xyz[1]) - ref_axial),
                "confidence": estimate.confidence,
                "cluster": cluster,
                "runtime_ms": estimate.latency_s * 1000.0,
            })
            if xyz is not None:
                unit_estimates[name][cluster].append(xyz)
        names = list(engines)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                xa, xb = xyzs[names[a]], xyzs[names[b]]
                if xa is None or xb is None:
                    continue
                method_pairs[f"{names[a]}|{names[b]}"].append(float(np.linalg.norm(np.subtract(xa, xb))))
    loc_summary = {}
    for method in LOCALIZATION_METHODS:
        selected = [r for r in loc_rows if r["method"] == method]
        disagreements = [r["axial_disagreement_um"] for r in selected if r["axial_disagreement_um"] is not None]
        loc_summary[method] = {
            "n": len(selected),
            "failures": sum(1 for r in selected if not r["available"]),
            "median_axial_disagreement_um": float(np.median(disagreements)) if disagreements else None,
            "p90_axial_disagreement_um": float(np.percentile(disagreements, 90)) if disagreements else None,
            "median_runtime_ms": float(np.median([r["runtime_ms"] for r in selected])) if selected else None,
        }
    unit_consistency = {}
    for method, clusters in unit_estimates.items():
        spreads = []
        for xyz_list in clusters.values():
            if len(xyz_list) < 2:
                continue
            arr = np.asarray(xyz_list, dtype=float)
            spreads.append(float(np.mean(np.linalg.norm(arr - arr.mean(axis=0), axis=1))))
        unit_consistency[method] = {
            "n_units_with_repeats": len(spreads),
            "median_within_unit_spread_um": float(np.median(spreads)) if spreads else None,
        }
    pair_summary = {pair: float(np.median(values)) for pair, values in method_pairs.items() if values}
    detected_loc = []
    for event in chosen["events"][:max_sorter_spikes]:
        snippet = extract_snippet(
            voltages, event.sample, chunk["array"], event_id=int(event.sample),
            sample_rate_hz=fs, peak_channel=chunk["array"].channel_ids()[event.channel]
            if event.channel < chunk["array"].n_channels else None,
        )
        for name, engine in engines.items():
            estimate = engine.estimate(snippet)
            detected_loc.append(name if estimate.available else f"{name}:fail")
    return {
        "t0_s": t0_s,
        "duration_s": duration_s,
        "pid": pid,
        "eid": str(chunk["eid"]),
        "native_sample_rate_hz": fs,
        "raw_samples": int(voltages.shape[0]),
        "raw_voltage_abs_p99_uv": float(np.percentile(np.abs(voltages), 99)),
        "sorter_spikes_on_selected_contacts": int(np.count_nonzero(mask)),
        "legacy_events": len(legacy_events),
        "legacy_vs_sorting_1ms": legacy_metrics,
        "sorter_aligned_snr_legacy": snr_legacy,
        "sorter_aligned_snr_improved": snr_improved,
        "detector_variants": variant_reports,
        "localization_sorter_triggered": loc_summary,
        "within_unit_consistency": unit_consistency,
        "median_method_disagreement_um": pair_summary,
        "detector_triggered_available": int(sum(1 for item in detected_loc if not item.endswith(":fail"))),
        "uncertainty": (
            "Sorter peak-channel depth is a proxy, not a neuron coordinate. "
            "Disagreement is not localization error. Sorting labels are incomplete."
        ),
        "sorting_revision": SORTING_REVISION,
    }


def run(windows: tuple[float, ...] = (50.0, 100.0, 200.0), duration_s: float = 1.0) -> dict:
    reports = [evaluate_window(t0, duration_s) for t0 in windows]
    # Cross-window: compare 4σ CAR recall without retuning thresholds per window.
    recalls = [r["detector_variants"]["car_4sigma"]["vs_sorting"]["1.0"]["recall"] for r in reports]
    legacy_recalls = [r["legacy_vs_sorting_1ms"]["recall"] for r in reports]
    return {
        "status": "ran",
        "pid": DEFAULT_PID,
        "windows_s": list(windows),
        "duration_s": duration_s,
        "legacy_recall_mean": float(np.mean(legacy_recalls)),
        "improved_car_4sigma_recall_mean": float(np.mean(recalls)),
        "improved_car_4sigma_recall_range": [float(np.min(recalls)), float(np.max(recalls))],
        "per_window": reports,
        "calibration_policy": "each window estimates MAD on itself; no labels used for thresholds",
        "cross_recording": "same insertion, disjoint 1 s AP windows; not an independent animal",
        "rtl": "not modified; comparison is Python-only",
    }


def main() -> None:
    target = Path("reports/ibl-detection-localization.json")
    try:
        report = run()
    except Exception as exc:
        report = {"status": "failed", "pid": DEFAULT_PID, "error": f"{type(exc).__name__}: {exc}"}
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "legacy_recall_mean", "improved_car_4sigma_recall_mean", "error")},
                     indent=2))


if __name__ == "__main__":
    main()
