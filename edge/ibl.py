"""Small raw IBL Neuropixels AP pilot with native geometry and sorted references.

Run with edge/requirements-ibl.txt installed. Sorted locations are comparison
references, not independently known neuron positions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy import signal

from edge.localize import LOCALIZATION_METHODS, AmplitudeLocalizer, BeamformLocalizer, TemplateLocalizer, get_localizer
from edge.localize.beamform import default_scan_grid_um
from edge.localize.template import default_training_xyz_um
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, ElectrodeSite
from scripts.detect_spikes import DetectorConfig, detect_candidates, estimate_noise, match_events, merge_channel_events
from scripts.filter_iir import IirConfig, design_sos
from scripts.real_detector import RealDetectorConfig, detect_real, evaluate_against_reference, sorter_aligned_snr

DEFAULT_PID = "4836a465-c691-4852-a0b1-dcd2b1ce38a1"
DEFAULT_EID = "caa5dddc-9290-4e27-9f5e-575ba3598614"
DEFAULT_PROBE = "probe00"
# Second public insertion: NYU-45 2021-07-19 probe00 (VISa, not the same mouse).
SECOND_PID = "1a60a6e1-da99-4d4e-a734-39b1d4544fad"
SORTING_REVISION = "2024-05-06"
TRIALS_REVISION = "2025-03-03"


def select_visp_channels(acronyms: np.ndarray, n_channels: int = 32) -> np.ndarray:
    selected = np.flatnonzero(np.char.startswith(np.asarray(acronyms).astype(str), "VISp"))
    if len(selected) < n_channels:
        selected = np.flatnonzero(np.char.startswith(np.asarray(acronyms).astype(str), "VIS"))
    if len(selected) < n_channels:
        raise ValueError(f"only {len(selected)} visual channels, need {n_channels}")
    return selected[:n_channels]


def probe_geometry(channel_ids: np.ndarray, local_coordinates_um: np.ndarray) -> ElectrodeArray:
    coordinates = np.asarray(local_coordinates_um, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("IBL localCoordinates must be [channel, lateral/axial]")
    ids = np.asarray(channel_ids, dtype=int)
    sites = tuple(ElectrodeSite(int(channel),
                                (float(coordinates[channel, 0]), float(coordinates[channel, 1]), 0.0))
                  for channel in ids)
    return ElectrodeArray(sites, "IBL native lateral/axial probe coordinates in micrometers")


def load_raw_chunk(pid: str = DEFAULT_PID, t0_s: float = 100.0, duration_s: float = 1.0,
                   n_channels: int = 32):
    """Stream native AP voltage plus sorting metadata for a time window."""
    from one.api import ONE
    from brainbox.io.spikeglx import Streamer
    from iblatlas.regions import BrainRegions

    one = ONE(base_url="https://openalyx.internationalbrainlab.org",
              username="intbrainlab", password="international", silent=True,
              cache_dir="data/neuropixels/ibl_cache")
    eid, probe = one.pid2eid(pid)
    collection = f"alf/{probe}/pykilosort"

    def dataset(name: str):
        return one.load_dataset(eid, name, collection=collection, revision=SORTING_REVISION)

    locations = np.asarray(dataset("channels.localCoordinates.npy"), dtype=float)
    acronyms = BrainRegions().id2acronym(dataset("channels.brainLocationIds_ccf_2017.npy"))
    selected = select_visp_channels(acronyms, n_channels)
    array = probe_geometry(selected, locations)
    raw_indices = np.asarray(dataset("channels.rawInd.npy"), dtype=int)[selected]
    streamer = Streamer(pid=pid, one=one, typ="ap", remove_cached=False)
    fs = float(streamer.fs)
    first, stop = int(round(t0_s * fs)), int(round((t0_s + duration_s) * fs))
    raw_v = np.asarray(streamer[first:stop, :streamer.nc - streamer.nsync], dtype=float)
    if raw_v.shape[0] != stop - first:
        raise ValueError("short raw AP stream")
    voltages_uv = raw_v[:, raw_indices] * 1e6
    if not np.isfinite(voltages_uv).all():
        raise ValueError("non-finite AP voltages")
    spike_times = np.asarray(dataset("spikes.times.npy"), dtype=float)
    spike_clusters = np.asarray(dataset("spikes.clusters.npy"), dtype=int)
    cluster_channels = np.asarray(dataset("clusters.channels.npy"), dtype=int)
    mask = (spike_times >= t0_s) & (spike_times < t0_s + duration_s)
    mask &= np.isin(cluster_channels[spike_clusters], selected)
    return {
        "one": one,
        "eid": eid,
        "probe": probe,
        "fs": fs,
        "locations": locations,
        "selected": selected,
        "array": array,
        "raw_indices": raw_indices,
        "voltages_uv": voltages_uv,
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "cluster_channels": cluster_channels,
        "sorter_mask": mask,
        "t0_s": t0_s,
        "duration_s": duration_s,
        "pid": pid,
    }


def _legacy_float_detector(voltages_uv: np.ndarray, fs: float):
    filtered_uv = signal.sosfilt(design_sos(IirConfig(sample_rate_hz=fs)), voltages_uv, axis=0)
    detector_config = DetectorConfig(sample_rate_hz=fs)
    baseline, noise = estimate_noise(filtered_uv)
    detected = merge_channel_events(detect_candidates(filtered_uv, baseline, noise, detector_config), detector_config)
    return filtered_uv, baseline, noise, detected, detector_config


def run_pilot(pid: str = DEFAULT_PID, t0_s: float = 100.0, duration_s: float = 1.0,
              n_channels: int = 32, max_spikes: int = 50) -> dict:
    chunk = load_raw_chunk(pid, t0_s, duration_s, n_channels)
    one, eid, probe, fs = chunk["one"], chunk["eid"], chunk["probe"], chunk["fs"]
    locations, selected, array = chunk["locations"], chunk["selected"], chunk["array"]
    raw_indices, voltages_uv = chunk["raw_indices"], chunk["voltages_uv"]
    spike_times, spike_clusters = chunk["spike_times"], chunk["spike_clusters"]
    cluster_channels, mask = chunk["cluster_channels"], chunk["sorter_mask"]
    spike_idx = np.flatnonzero(mask)[:max_spikes]
    filtered_uv, baseline, noise, detected, detector_config = _legacy_float_detector(voltages_uv, fs)
    _, sorting_match = match_events(detected, spike_times[mask] - t0_s, detector_config)
    improved_cfg = RealDetectorConfig(sample_rate_hz=fs)
    improved = detect_real(voltages_uv, improved_cfg)
    improved_match = evaluate_against_reference(improved["events"], spike_times[mask] - t0_s, improved_cfg)
    snr = sorter_aligned_snr(improved["filtered_uv"], improved["baseline"], improved["noise_sigma"],
                             spike_times[mask] - t0_s, fs)
    y_shift = float(np.min(array.xyz_um()[:, 1]))
    training = default_training_xyz_um(n_channels) + np.array([0.0, y_shift, 0.0])
    scan = default_scan_grid_um(n_channels) + np.array([0.0, y_shift, 0.0])
    methods = {
        "amplitude": AmplitudeLocalizer(),
        "template": TemplateLocalizer(array=array, training_xyz_um=training),
        "beamform": BeamformLocalizer(array=array, grid_xyz_um=scan),
        "robust_scan": get_localizer("robust_scan", array=array, grid_xyz_um=scan),
    }
    differences: dict[str, list[float]] = {name: [] for name in methods}
    for i, index in enumerate(spike_idx):
        sample = int(round((spike_times[index] - t0_s) * fs))
        peak_channel = int(cluster_channels[spike_clusters[index]])
        snippet = extract_snippet(voltages_uv, sample, array, event_id=i,
                                  sample_rate_hz=fs, peak_channel=peak_channel)
        ref_axial = float(locations[peak_channel, 1])
        for name, engine in methods.items():
            estimate = engine.estimate(snippet)
            if estimate.available and estimate.xyz_um is not None:
                differences[name].append(abs(float(estimate.xyz_um[1]) - ref_axial))
    trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf",
                              revision=TRIALS_REVISION)
    return {
        "status": "ran", "pid": pid, "eid": str(eid), "probe": probe,
        "t0_s": t0_s, "duration_s": duration_s, "native_sample_rate_hz": fs,
        "selected_channel_ids": selected.tolist(),
        "selected_raw_indices": raw_indices.tolist(),
        "selected_local_coordinates_um": array.xyz_um().tolist(),
        "raw_kind": "calibrated AP voltage converted V to µV; sync channels excluded",
        "raw_samples": int(voltages_uv.shape[0]),
        "raw_voltage_abs_p99_uv": float(np.percentile(np.abs(voltages_uv), 99)),
        "sorting_revision": SORTING_REVISION, "trials_revision": TRIALS_REVISION,
        "sorted_spikes_in_subset": len(spike_idx),
        "sorted_spikes_in_chunk_for_selected_peak_channels": int(np.count_nonzero(mask)),
        "float_detector_events": len(detected),
        "float_detector_vs_sorting_1ms": sorting_match,
        "sorter_aligned_snr_after_car": snr,
        "improved_python_detector": {
            "events": improved["merged_detections"],
            "channel_candidates": improved["channel_candidates"],
            "artifact_rejected_candidates": improved["artifact_rejected_candidates"],
            "noise_sigma_uv": improved["noise_sigma_uv"],
            "vs_sorting_ms": improved_match,
            "config": improved["config"],
        },
        "detector_caveat": "30 kHz float voltage filter/detector versus sorter spikes on selected peak channels; no ADC quantization or RTL equivalence, and sorting is an incomplete reference.",
        "rtl_status": "unchanged; real-recording CAR, 4σ calibration, and artifact veto are Python-only",
        "trial_columns": list(trials.columns),
        "median_axial_disagreement_to_sorter_peak_channel_um": {
            name: float(np.median(values)) if values else None for name, values in differences.items()},
        "uncertainty": "Sorter peak channel is a proxy, not an independently established neuron position; do not interpret disagreement as localization accuracy.",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("one-api", "ibllib", "ibl-neuropixel")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", default=DEFAULT_PID)
    parser.add_argument("--t0", type=float, default=100.0)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=Path("reports/ibl-pilot.json"))
    args = parser.parse_args()
    try:
        report = run_pilot(args.pid, args.t0, args.seconds)
    except Exception as exc:
        report = {"status": "failed", "pid": args.pid, "t0_s": args.t0,
                  "duration_s": args.seconds, "error": f"{type(exc).__name__}: {exc}"}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
