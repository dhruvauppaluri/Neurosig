"""Phase 9 decoding revisit: counts vs electrode vs original vs P9 (frozen) localization, matched controls.

Uses the Phase 8 session loader/CV; adds (a) P9 locations (frozen weighted_scan, baseline+CAR conditioning, lambda=20 line source)
with an uncertainty feature, (b) a redundancy analysis: out-of-fold R^2 of the spatial features given the counts,
(c) a position-only decoder. Nothing is tuned; folds, models and hyperparameters are fixed.
Run: PYTHONPATH=. data/ibl-venv/bin/python -m edge.decode.phase9_ablation
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from edge.decode.model import predict_labels, train
from edge.decode.phase8_ablation import (cv_predictions, load_session, paired_boot, pick_units, spatial_features,
                                         trial_table, unit_locations)
from edge.ibl import probe_geometry
from edge.phase9.configs import P9_HASH, p9_factory

SESSIONS = {"A": "4836a465-c691-4852-a0b1-dcd2b1ce38a1", "B": "1a60a6e1-da99-4d4e-a734-39b1d4544fad",
            "C": "7a620688-66cb-44d3-b79b-ccac1c8ba23e", "D": "b114e7e0-92b3-43bb-9529-53860e52606d",
            "E": "c5b9e063-f640-4936-b851-f7602cb6659b", "F": "5135e93f-2f1f-4301-9532-b5ad62548c49",
            "G": "18d316bf-d322-4c5c-814e-a58147f7bf5f", "H": "730770d6-617a-4ada-95db-a48521befda5"}


def p9_unit_locations(s, units):
    from edge.types import Snippet
    xyz, unc = [], []
    for u in units:
        ch = tuple(int(c) for c in s["waveform_channels"][u])
        array = probe_geometry(np.asarray(ch), s["coords"])
        sn = Snippet(int(u), 41, 0, 30000.0, ch, s["waveforms"][u] * 1e6, array.xyz_um(), int(s["cluster_channel"][u]))
        e = p9_factory(array)["weighted_scan"].estimate(sn)
        xyz.append(e.xyz_um if e.available and e.xyz_um else tuple(array.xyz_um().mean(axis=0)))
        unc.append(1.0 - float(e.confidence or 0.0))
    return np.array(xyz, float), np.array(unc, float)


def oof_r2(counts, feats, labels, seed=0):
    """How much of each spatial feature column is explained by the counts (ridge, 5-fold out-of-fold R^2)."""
    n = len(counts); rng = np.random.default_rng(seed); fold = rng.permutation(n) % 5
    X = np.hstack([counts, np.ones((n, 1))]); r2 = []
    for j in range(feats.shape[1]):
        y = feats[:, j]
        if np.std(y) < 1e-9:
            continue
        pred = np.zeros(n)
        for f in range(5):
            tr, te = fold != f, fold == f
            w = np.linalg.solve(X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1]), X[tr].T @ y[tr]); pred[te] = X[te] @ w
        r2.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(np.median(r2)), float(np.min(r2))


def run_session(label, pid, seed=20269400):
    s = load_session(pid)
    units, region = pick_units(s)
    if len(units) < 24:
        return {"session": label, "status": "skipped", "reason": f"only {len(units)} units"}
    counts, labels = trial_table(s, units)
    orig, new8, unc8, chi8, site = unit_locations(s, units)
    p9, p9u = p9_unit_locations(s, units)
    ymin = float(min(site[:, 1].min(), orig[:, 1].min(), p9[:, 1].min()) - 20); ymax = float(max(site[:, 1].max(), orig[:, 1].max(), p9[:, 1].max()) + 20)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(units))
    F = {"1_counts": counts, "2_counts+electrode": spatial_features(counts, site, ymin, ymax),
         "3_counts+orig_loc": spatial_features(counts, orig, ymin, ymax),
         "4_counts+p9_loc": spatial_features(counts, p9, ymin, ymax),
         "5_counts+p9_loc+unc": spatial_features(counts, p9, ymin, ymax, extra=p9u),
         "ctrl_p9_shuffled": spatial_features(counts, p9[perm], ymin, ymax),
         "mlp_counts": counts, "mlp_p9+unc": spatial_features(counts, p9, ymin, ymax, extra=p9u)}
    pos_only = spatial_features(np.zeros_like(counts), p9, ymin, ymax)    # placeholder: positions without counts carry no trial info
    corr = cv_predictions(F, labels, seed)
    acc = {k: float(v.mean()) for k, v in corr.items()}
    pairs = {"p9_vs_counts": ("1_counts", "4_counts+p9_loc"), "p9_vs_orig": ("3_counts+orig_loc", "4_counts+p9_loc"),
             "p9unc_vs_counts": ("1_counts", "5_counts+p9_loc+unc"), "p9_vs_shuffled": ("ctrl_p9_shuffled", "4_counts+p9_loc"),
             "orig_vs_counts": ("1_counts", "3_counts+orig_loc"), "electrode_vs_counts": ("1_counts", "2_counts+electrode"),
             "mlp_p9_vs_mlp_counts": ("mlp_counts", "mlp_p9+unc")}
    paired = {k: paired_boot(corr[a], corr[b], seed + i) for i, (k, (a, b)) in enumerate(pairs.items())}
    # class-specific recall and calibration proxy (accuracy on confident-vs-unconfident trials is not available for fixed linear sign decoder)
    r2_med, r2_min = oof_r2(counts, F["4_counts+p9_loc"][:, counts.shape[1]:], labels)
    return {"session": label, "pid": pid, "status": "ran", "region_prefix": region, "n_trials": int(len(labels)),
            "majority_accuracy": float(max(np.bincount(labels)) / len(labels)), "accuracy": acc, "paired": paired,
            "spatial_features_explained_by_counts_oof_R2": {"median": r2_med, "min": r2_min},
            "p9_vs_orig_median_location_shift_um": float(np.median(np.linalg.norm(p9[:, :2] - orig[:, :2], axis=1))),
            "p9_uncertainty_median": float(np.median(p9u))}


def main():
    out = []
    for lab, pid in SESSIONS.items():
        t = time.time()
        try:
            out.append(run_session(lab, pid))
        except Exception as exc:  # noqa: BLE001
            out.append({"session": lab, "pid": pid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(lab, out[-1].get("status"), out[-1].get("accuracy") or out[-1].get("error") or out[-1].get("reason"), f"{time.time()-t:.0f}s", flush=True)
    Path("reports/phase9/decode_ablation.json").write_text(json.dumps({"p9_hash": P9_HASH, "sessions": out}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
