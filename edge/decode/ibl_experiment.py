"""Three-model visual-side decoder on one IBL sorted Neuropixels session.

Model C uses locations estimated from provided average cluster waveforms. The
independent raw AP chunk is assessed by edge.ibl; these trial features are
sorted spikes and processed waveforms, not continuous raw voltage.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from edge.decode.experiment_v2 import classification
from edge.decode.features import stack_spatial
from edge.decode.model import predict_labels, train
from edge.ibl import DEFAULT_PID, SORTING_REVISION, TRIALS_REVISION, probe_geometry
from edge.localize.amplitude import AmplitudeLocalizer
from edge.types import Snippet


def run() -> dict:
    from one.api import ONE
    from iblatlas.regions import BrainRegions

    one = ONE(base_url="https://openalyx.internationalbrainlab.org",
              username="intbrainlab", password="international", silent=True,
              cache_dir="data/neuropixels/ibl_cache")
    eid, probe = one.pid2eid(DEFAULT_PID)
    collection = f"alf/{probe}/pykilosort"
    def dataset(name: str):
        return one.load_dataset(eid, name, collection=collection, revision=SORTING_REVISION)
    coords = np.asarray(dataset("channels.localCoordinates.npy"), dtype=float)
    acronyms = BrainRegions().id2acronym(dataset("channels.brainLocationIds_ccf_2017.npy"))
    cluster_channels = np.asarray(dataset("clusters.channels.npy"), dtype=int)
    waveforms = np.asarray(dataset("clusters.waveforms.npy"), dtype=float)
    waveform_channels = np.asarray(dataset("clusters.waveformsChannels.npy"), dtype=int)
    selected = np.flatnonzero(np.char.startswith(acronyms[cluster_channels].astype(str), "VISp"))[:32]
    if len(selected) != 32:
        raise ValueError("fewer than 32 VISp sorted units")
    if waveforms.shape[0] <= int(np.max(selected)):
        raise ValueError("cluster waveform table does not match channels")
    estimated = []
    geometry = []
    engine = AmplitudeLocalizer()
    for cluster in selected:
        channel_ids = tuple(int(c) for c in waveform_channels[cluster])
        array = probe_geometry(np.asarray(channel_ids), coords)
        waveform_uv = waveforms[cluster] * 1e6
        if not np.isfinite(waveform_uv).all():
            raise ValueError("non-finite cluster waveform")
        peak = int(cluster_channels[cluster])
        snippet = Snippet(event_id=int(cluster), t_sample=41, t0_sample=0,
                          sample_rate_hz=30000.0, channel_ids=channel_ids,
                          voltages_uv=waveform_uv, electrode_xyz_um=array.xyz_um(),
                          peak_channel=peak)
        estimate = engine.estimate(snippet)
        if not estimate.available or estimate.xyz_um is None:
            raise ValueError("unavailable waveform localization")
        estimated.append(estimate.xyz_um)
        geometry.append(tuple(float(v) for v in coords[peak]) + (0.0,))
    spikes_t = np.asarray(dataset("spikes.times.npy"), dtype=float)
    spikes_c = np.asarray(dataset("spikes.clusters.npy"), dtype=int)
    if len(spikes_t) != len(spikes_c):
        raise ValueError("sorted spike arrays differ in length")
    trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf",
                              revision=TRIALS_REVISION)
    left = np.nan_to_num(np.asarray(trials["contrastLeft"], dtype=float), nan=0.0)
    right = np.nan_to_num(np.asarray(trials["contrastRight"], dtype=float), nan=0.0)
    onset = np.asarray(trials["stimOn_times"], dtype=float)
    eligible = np.flatnonzero(np.isfinite(onset) & ((left > 0) ^ (right > 0)))
    labels = (right[eligible] > 0).astype(np.int8)
    counts = np.zeros((len(eligible), 32), dtype=float)
    for row, trial in enumerate(eligible):
        start = int(np.searchsorted(spikes_t, onset[trial] + 0.05))
        stop = int(np.searchsorted(spikes_t, onset[trial] + 0.25))
        values = spikes_c[start:stop]
        counts[row] = np.array([np.count_nonzero(values == cluster) for cluster in selected])
    site_xyz = np.asarray(geometry, dtype=float)
    estimated_xyz = np.asarray(estimated, dtype=float)
    def spatial(points: np.ndarray) -> np.ndarray:
        return stack_spatial(counts, [np.repeat(points, counts[i].astype(int), axis=0)
                                      for i in range(len(counts))])
    inputs = {"A_counts": counts, "B_electrode": spatial(site_xyz),
              "C_estimated": spatial(estimated_xyz)}
    reports = []
    for seed in (20261103, 20261104, 20261105):
        rng = np.random.default_rng(seed)
        train_ids, test_ids = [], []
        for label in (0, 1):
            ids = rng.permutation(np.flatnonzero(labels == label))
            split = int(0.7 * len(ids))
            train_ids.extend(ids[:split]); test_ids.extend(ids[split:])
        train_ids = np.asarray(train_ids, dtype=int)
        test_ids = np.asarray(test_ids, dtype=int)
        models = {}
        c_model = None
        for name, x in inputs.items():
            weights, bias = train(x[train_ids], labels[train_ids].astype(float))
            if name == "C_estimated":
                c_model = (weights, bias)
            start = time.perf_counter()
            pred = predict_labels(x[test_ids], weights, bias)
            elapsed = time.perf_counter() - start
            models[name] = {**classification(labels[test_ids], pred),
                            "n_features": int(x.shape[1]), "model_bytes_float64": int(weights.nbytes + 8),
                            "inference_us_per_trial": elapsed * 1e6 / len(test_ids)}
        assert c_model is not None
        sensitivity = {}
        for sigma in (0.0, 20.0, 50.0, 100.0):
            perturbed = estimated_xyz + np.random.default_rng(seed + int(sigma) + 991).normal(
                0.0, sigma, size=estimated_xyz.shape)
            x_perturbed = spatial(perturbed)
            sensitivity[str(int(sigma))] = classification(
                labels[test_ids], predict_labels(x_perturbed[test_ids], *c_model))["accuracy"]
        reports.append({"seed": seed, "train": len(train_ids), "test": len(test_ids),
                        "majority_test_accuracy": float(max(np.bincount(labels[test_ids])) / len(test_ids)),
                        "models": models, "estimate_perturbation_um": sensitivity})
    aggregate = {name: {"mean_accuracy": float(np.mean([r["models"][name]["accuracy"] for r in reports])),
                        "min_accuracy": float(np.min([r["models"][name]["accuracy"] for r in reports])),
                        "max_accuracy": float(np.max([r["models"][name]["accuracy"] for r in reports]))}
                 for name in inputs}
    return {"status": "ran", "pid": DEFAULT_PID, "eid": str(eid), "probe": probe,
            "sorting_revision": SORTING_REVISION, "trials_revision": TRIALS_REVISION,
            "task": "left versus right visual stimulus, exactly one side with positive contrast",
            "window": "50-250 ms after stimOn_times", "eligible_trials": len(eligible),
            "selected_cluster_ids": selected.tolist(), "feature_source": "sorted spikes and processed average cluster waveforms",
            "ground_truth_source_coordinates_used": False,
            "models": reports, "aggregate": aggregate,
            "limitation": "One session; cluster mean waveforms were computed using the entire recording without labels, so location features are transductive. Raw AP pilot is separate. No exact neuron-position truth or RTL inference claimed."}


def main() -> None:
    target = Path("reports/ibl-decode.json")
    try:
        report = run()
    except Exception as exc:
        report = {"status": "failed", "pid": DEFAULT_PID,
                  "error": f"{type(exc).__name__}: {exc}"}
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("aggregate", report), indent=2))


if __name__ == "__main__":
    main()
