"""Phase 8 decoding ablation: does better localization add information for left/right visual decoding?

Per IBL insertion (network via openalyx, run in data/ibl-venv):
  units  : up to 32 sorted clusters from the most populated brain-region prefix
  counts : spike counts 50-250 ms after stimOn, trials with exactly one side of positive contrast
  locations: per-unit position from the cluster mean waveform with (a) the original robust_scan and
             (b) the Phase 8 K=1 space-time fit (+ uncertainty). No trial labels are used to place units.
Evaluation: repeated stratified 5-fold CV (4 repeats), fixed hyperparameters (nothing tuned), same folds for every
feature set. A shuffled-position control keeps feature dimension identical so that capacity alone cannot
explain a gain. Paired bootstrap over trials per session, then a sign/bootstrap summary over sessions.

Usage: PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase8_ablation --pids <pid8,...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from edge.decode.features import y_occupancy
from edge.decode.model import predict_labels, predict_mlp, train, train_mlp
from edge.ibl import SORTING_REVISION, TRIALS_REVISION, probe_geometry

PIDS = {
    "A": "4836a465-c691-4852-a0b1-dcd2b1ce38a1", "B": "1a60a6e1-da99-4d4e-a734-39b1d4544fad",
}
N_UNITS = 32
N_BINS = 8


def load_session(pid: str):
    from iblatlas.regions import BrainRegions
    from one.api import ONE
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
              password="international", silent=True, cache_dir="data/neuropixels/ibl_cache")
    eid, probe = one.pid2eid(pid)
    coll = f"alf/{probe}/pykilosort"

    def ds(name):
        return one.load_dataset(eid, name, collection=coll, revision=SORTING_REVISION)
    coords = np.asarray(ds("channels.localCoordinates.npy"), float)
    acr = BrainRegions().id2acronym(ds("channels.brainLocationIds_ccf_2017.npy")).astype(str)
    cch = np.asarray(ds("clusters.channels.npy"), int)
    wf = np.asarray(ds("clusters.waveforms.npy"), float)
    wch = np.asarray(ds("clusters.waveformsChannels.npy"), int)
    st = np.asarray(ds("spikes.times.npy"), float)
    sc = np.asarray(ds("spikes.clusters.npy"), int)
    trials = one.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf", revision=TRIALS_REVISION)
    return dict(eid=str(eid), probe=probe, coords=coords, acronyms=acr, cluster_channel=cch, waveforms=wf,
                waveform_channels=wch, spike_times=st, spike_clusters=sc, trials=trials)


def pick_units(s) -> tuple[np.ndarray, str]:
    names = np.char.strip(s["acronyms"][s["cluster_channel"]].astype(str))
    prefixes = np.array([n[:3] for n in names])
    vals, cnt = np.unique(prefixes, return_counts=True)
    best = vals[int(np.argmax(cnt))]
    idx = np.flatnonzero((prefixes == best) & (s["waveforms"].reshape(len(prefixes), -1).std(axis=1) > 0))
    return idx[:N_UNITS], str(best)


def trial_table(s, units):
    tr = s["trials"]
    left = np.nan_to_num(np.asarray(tr["contrastLeft"], float), nan=0.0)
    right = np.nan_to_num(np.asarray(tr["contrastRight"], float), nan=0.0)
    onset = np.asarray(tr["stimOn_times"], float)
    ok = np.flatnonzero(np.isfinite(onset) & ((left > 0) ^ (right > 0)))
    labels = (right[ok] > 0).astype(int)
    st, sc = s["spike_times"], s["spike_clusters"]
    counts = np.zeros((len(ok), len(units)))
    for r, t in enumerate(ok):
        a, b = np.searchsorted(st, onset[t] + 0.05), np.searchsorted(st, onset[t] + 0.25)
        v = sc[a:b]
        counts[r] = [np.count_nonzero(v == u) for u in units]
    return counts, labels


def unit_locations(s, units):
    from edge.localize.beamform import default_scan_grid_um
    from edge.localize import get_localizer
    from edge.phase8.pipeline import Phase8Localizer
    from edge.types import Snippet
    orig, new, unc, chi, site = [], [], [], [], []
    for u in units:
        ch = tuple(int(c) for c in s["waveform_channels"][u])
        array = probe_geometry(np.asarray(ch), s["coords"])
        wf = s["waveforms"][u] * 1e6
        sn = Snippet(int(u), 41, 0, 30000.0, ch, wf, array.xyz_um(), int(s["cluster_channel"][u]))
        shift = np.array([0.0, float(array.xyz_um()[:, 1].min()), 0.0])
        eng = get_localizer("robust_scan", array=array, grid_xyz_um=default_scan_grid_um(array.n_channels) + shift,
                            model="line_source")
        e = eng.estimate(sn)
        orig.append(e.xyz_um if e.available and e.xyz_um else tuple(array.xyz_um().mean(axis=0)))
        p8 = Phase8Localizer(array, model="line_source")
        fit, Wt, sigma = p8.analyze(sn)
        src = fit.sources1[0]
        cov = p8.fitter.covariance(np.array([[src.x, src.y]]), Wt, sigma)
        new.append((src.x, src.y, 0.0))
        unc.append(float(np.sqrt(max(np.trace(cov), 0.0))))
        chi.append(float(fit.sse1 / max(Wt.size - 2 - Wt.shape[1], 1)))
        site.append(tuple(s["coords"][int(s["cluster_channel"][u])]) + (0.0,))
    return (np.array(orig, float), np.array(new, float), np.array(unc, float), np.array(chi, float),
            np.array(site, float))


def spatial_features(counts, pos, ymin, ymax, extra=None):
    """counts + count-weighted y occupancy (N_BINS) + count-weighted mean xyz (+ optional per-trial extra)."""
    n = counts.shape[0]
    out = np.zeros((n, counts.shape[1] + N_BINS + 3 + (0 if extra is None else 1)))
    for i in range(n):
        w = counts[i]
        reps = np.repeat(pos, w.astype(int), axis=0)
        occ = y_occupancy(reps[:, 1] if len(reps) else np.zeros(0), n_bins=N_BINS, y_min=ymin, y_max=ymax)
        mean = reps.mean(axis=0) if len(reps) else np.zeros(3)
        row = [counts[i], occ, mean]
        if extra is not None:
            row.append([float((w * extra).sum() / max(w.sum(), 1.0))])
        out[i] = np.concatenate(row)
    return out


def cv_predictions(feature_sets: dict, labels, seed: int, n_rep: int = 4, n_fold: int = 5):
    """Out-of-fold predictions, returns {name: [n_rep, n_trials] of correctness} with identical folds."""
    n = len(labels)
    corr = {k: np.zeros((n_rep, n)) for k in feature_sets}
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        fold = np.zeros(n, int)
        for lab in (0, 1):
            idx = rng.permutation(np.flatnonzero(labels == lab))
            fold[idx] = np.arange(len(idx)) % n_fold
        for f in range(n_fold):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            for name, x in feature_sets.items():
                if name.startswith("mlp_"):
                    packed = train_mlp(x[tr], labels[tr].astype(float), seed=seed + rep * 10 + f)
                    pred = predict_mlp(x[te], *packed)
                else:
                    w, b = train(x[tr], labels[tr].astype(float))
                    pred = predict_labels(x[te], w, b)
                corr[name][rep, te] = (np.asarray(pred).astype(int) == labels[te]).astype(float)
    return corr


def paired_boot(c_a, c_b, seed, n_boot=2000):
    a, b = c_a.mean(axis=0), c_b.mean(axis=0)   # per-trial mean correctness over repeats
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (b[idx] - a[idx]).mean(axis=1)
    return {"delta": float((b - a).mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def run_session(label: str, pid: str, seed: int = 20268400) -> dict:
    s = load_session(pid)
    units, region = pick_units(s)
    if len(units) < 24:
        return {"session": label, "pid": pid, "status": "skipped", "reason": f"only {len(units)} units in top region"}
    counts, labels = trial_table(s, units)
    if len(labels) < 80 or labels.min() == labels.max():
        return {"session": label, "pid": pid, "status": "skipped", "reason": f"{len(labels)} eligible trials"}
    keep = counts.sum(axis=0) > 0
    orig, new, unc, chi, site = unit_locations(s, units)
    ymin = float(min(site[:, 1].min(), orig[:, 1].min(), new[:, 1].min()) - 20)
    ymax = float(max(site[:, 1].max(), orig[:, 1].max(), new[:, 1].max()) + 20)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(units))
    feats = {
        "1_counts": counts,
        "2_counts+electrode": spatial_features(counts, site, ymin, ymax),
        "3_counts+orig_loc": spatial_features(counts, orig, ymin, ymax),
        "4_counts+new_loc": spatial_features(counts, new, ymin, ymax),
        "5_counts+new_loc+unc": spatial_features(counts, new, ymin, ymax, extra=unc),
        "ctrl_new_loc_shuffled": spatial_features(counts, new[perm], ymin, ymax),
        "mlp_counts": counts,
        "mlp_new_loc+unc": spatial_features(counts, new, ymin, ymax, extra=unc),
    }
    corr = cv_predictions(feats, labels, seed)
    acc = {k: float(v.mean()) for k, v in corr.items()}
    maj = float(max(np.bincount(labels)) / len(labels))
    pairs = {"new_loc_vs_counts": ("1_counts", "4_counts+new_loc"),
             "new_loc_vs_orig_loc": ("3_counts+orig_loc", "4_counts+new_loc"),
             "new_loc+unc_vs_counts": ("1_counts", "5_counts+new_loc+unc"),
             "new_loc_vs_shuffled_control": ("ctrl_new_loc_shuffled", "4_counts+new_loc"),
             "orig_loc_vs_counts": ("1_counts", "3_counts+orig_loc"),
             "electrode_vs_counts": ("1_counts", "2_counts+electrode"),
             "mlp_new_vs_mlp_counts": ("mlp_counts", "mlp_new_loc+unc")}
    paired = {k: paired_boot(corr[a], corr[b], seed + i) for i, (k, (a, b)) in enumerate(pairs.items())}
    return {"session": label, "pid": pid, "status": "ran", "eid": s["eid"], "region_prefix": region,
            "n_units": int(len(units)), "n_trials": int(len(labels)), "majority_accuracy": maj,
            "accuracy": acc, "paired": paired,
            "loc_shift_um": {"orig_vs_new_median": float(np.median(np.linalg.norm(orig[:, :2] - new[:, :2], axis=1)))},
            "unit_uncertainty_um_median": float(np.median(unc)), "unit_chi2_red_median": float(np.median(chi))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True, help="comma-separated LABEL=pid pairs or labels from known sessions")
    ap.add_argument("--out", default="reports/phase8/decode_ablation.json")
    a = ap.parse_args()
    out = []
    for item in a.pids.split(","):
        label, pid = item.split("=") if "=" in item else (item, PIDS[item])
        t = time.time()
        try:
            out.append(run_session(label, pid))
        except Exception as exc:  # noqa: BLE001
            out.append({"session": label, "pid": pid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(label, out[-1].get("status"), out[-1].get("accuracy") or out[-1].get("error") or out[-1].get("reason"),
              f"{time.time() - t:.0f}s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
