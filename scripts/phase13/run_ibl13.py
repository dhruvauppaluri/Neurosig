"""Phase 13 IBL visual-decoding experiments (A-D and Section 11 perturbations). Usage: run_ibl13.py dev|select|final

Splits (session-disjoint, fixed before evaluation): dev A-D | select E-L | final M-S (never decoded before in any
phase). Sessions need >= 20 visual units and >= 150 usable trials, otherwise 'skipped' is reported.

Trials: finite stimOn; window 40-120 ms after stimOn (2 bins of 40 ms). Dev finding (sessions A-C): the first wheel
movement falls before +200 ms on 75-98% of trials (and often before stimOn), so movement exclusion cannot be the main
analysis. Main analysis: all trials. Movement controls: T2 on 'still' trials only (no first movement within -200..+120
ms of stimOn), choice decoding, T2 accuracy on error trials, and the pre-stimulus window. Hemisphere from the probe's median CCF ML coordinate: the
contralateral stimulus is the one on the side opposite the probe.
Tasks: T1 presence (contrast 0 vs >= 0.25 on either side), T2 side (contralateral vs ipsilateral, contrast > 0),
T3 contralateral contrast level (0 / 6.25 / 12.5 / 25 / 100 %, 5 classes; ipsilateral-stimulus trials excluded),
T4 continuous contralateral log contrast (ridge), plus controls: T2 on the -200..0 ms pre-stimulus window (block-prior
leakage), T2 with permuted labels, choice decoding (reference), and T2 accuracy on error trials (stimulus vs choice).
Representations (identical trials and folds): counts, tcounts (4 x 40 ms), total, umap (counts placed at sorter unit
depth, 40 um bins), pmap (peak-channel depth = electrode coordinate), umap_shuf (unit depths permuted), smap
(per-spike sorter depth), stmap (per-spike depth x 4 time bins), counts_rp (counts randomly projected to umap's
dimension), counts+umap and counts+umap_shuf (matched capacity), smap with Gaussian depth noise (Section 11).
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase13 import decode as D
from edge.phase13.ibl_data import load

SPLITS = {"dev": "ABCD", "select": "EFGHIJKL", "final": "MNOPQRS"}
W0, W1, NB = 0.04, 0.12, 2
EDGE_UM = 40.0
NOISE = (10.0, 25.0, 50.0, 100.0, 200.0)
CONTRASTS = (0.0, 0.0625, 0.125, 0.25, 1.0)


def session(key):
    d = load(key)
    on = d["trial_stimOn_times"]; cl = np.nan_to_num(d["trial_contrastLeft"]); cr = np.nan_to_num(d["trial_contrastRight"])
    mv = d.get("trial_firstMovement_times", np.full(len(on), np.nan)); rt = mv - on
    ok = np.isfinite(on)
    still_all = ~(np.isfinite(rt) & (rt > -0.2) & (rt < 0.12))
    n_excl_move = int(np.sum(ok & ~still_all))
    idx = np.flatnonzero(ok)
    right_hemi = float(d["hemisphere_ml_um"]) > 0
    contra = cl if right_hemi else cr; ipsi = cr if right_hemi else cl
    units = np.asarray(d["vis_clusters"], int)
    rate = np.array([np.sum(d["spike_clusters"] == u) for u in units]) / max(float(d["spike_times"][-1] - d["spike_times"][0]), 1.0)
    keep = rate >= 0.5
    units = units[keep]
    udep = np.asarray(d["cluster_depth"], float)[keep]; pdep = np.asarray(d["cluster_peak_depth"], float)[keep]
    C, per = D.bin_spikes(d["spike_times"], d["spike_clusters"], on[idx], units, W0, W1, NB, extra=d["spike_depths"].astype(float))
    Cpre, _ = D.bin_spikes(d["spike_times"], d["spike_clusters"], on[idx], units, -0.2, 0.0, 1)
    return dict(d=d, idx=idx, still=still_all[idx], contra=contra[idx], ipsi=ipsi[idx], choice=d["trial_choice"][idx], fb=d["trial_feedbackType"][idx],
                units=units, udep=udep, pdep=pdep, C=C, per=per, Cpre=Cpre[:, :, 0], n_excl_move=n_excl_move, right_hemi=right_hemi)


def representations(s, rng):
    C = s["C"]; n = len(C)
    lo = np.floor(min(s["udep"].min(), s["pdep"].min()) / EDGE_UM) * EDGE_UM
    hi = np.ceil(max(s["udep"].max(), s["pdep"].max()) / EDGE_UM + 1) * EDGE_UM
    edges = np.arange(lo, hi + 1, EDGE_UM)
    R = {"counts": C.sum(2), "tcounts": C.reshape(n, -1), "total": C.sum((1, 2))[:, None]}
    R["umap"] = D.unit_depth_maps(C, s["udep"], edges)
    R["pmap"] = D.unit_depth_maps(C, s["pdep"], edges)
    R["umap_shuf"] = D.unit_depth_maps(C, rng.permutation(s["udep"]), edges)
    R["smap"] = D.spike_depth_maps(s["per"], n, edges, NB, False)
    R["stmap"] = D.spike_depth_maps(s["per"], n, edges, NB, True)
    R["counts_rp"] = D.random_projection(R["counts"], R["umap"].shape[1], seed=1)
    R["counts+umap"] = np.hstack([R["counts"], R["umap"]]); R["counts+umap_shuf"] = np.hstack([R["counts"], R["umap_shuf"]])
    for sg in NOISE:
        R[f"smap_noise{int(sg)}"] = D.spike_depth_maps(s["per"], n, edges, NB, False, noise_um=sg, rng=rng)
    return R, edges


def tasks(s):
    c, ip = s["contra"], s["ipsi"]
    T = {}
    m = (np.maximum(c, ip) == 0) | (np.maximum(c, ip) >= 0.25)
    T["T1_presence"] = (m, (np.maximum(c, ip) >= 0.25).astype(int))
    m = (c > 0) ^ (ip > 0)
    T["T2_side"] = (m, (c > 0).astype(int))
    m = ip == 0
    lvl = np.array([int(np.argmin(np.abs(np.array(CONTRASTS) - x))) for x in c])
    T["T3_contrast5"] = (m, lvl)
    T["T5_choice"] = (np.isin(s["choice"], (-1, 1)), (s["choice"] > 0).astype(int))
    return T


def run_session(key, seed=20271501):
    s = session(key)
    if len(s["units"]) < 20 or len(s["idx"]) < 150:
        return {"status": "skipped", "n_units": int(len(s["units"])), "n_trials": int(len(s["idx"]))}
    rng = np.random.default_rng(seed)
    R, edges = representations(s, rng)
    out = {"status": "ok", "n_units": int(len(s["units"])), "n_trials_used": int(len(s["idx"])), "n_moving_trials": s["n_excl_move"],
           "right_hemisphere": s["right_hemi"], "depth_bins": int(len(edges) - 1), "tasks": {}}
    lr_reps = list(R)
    mlp_reps = ("counts", "tcounts", "umap", "stmap")
    for tname, (m, y) in tasks(s).items():
        ii = np.flatnonzero(m); yy = y[ii]
        if len(ii) < 60 or len(np.unique(yy)) < 2 or np.min(np.unique(yy, return_counts=True)[1]) < 5:
            out["tasks"][tname] = {"status": "insufficient"}; continue
        folds = D.blocked_folds(len(ii))
        res, preds = {}, {}
        reps = lr_reps if tname != "T5_choice" else ["counts", "tcounts", "umap", "stmap"]
        for r in reps:
            p, P, cls = D.oof_classify(R[r][ii], yy, folds, "lr", seed)
            res[f"lr:{r}"] = D.clf_metrics(yy, p, P, cls); preds[f"lr:{r}"] = p
        if tname in ("T2_side", "T3_contrast5"):
            for r in mlp_reps:
                p, P, cls = D.oof_classify(R[r][ii], yy, folds, "mlp", seed)
                res[f"mlp:{r}"] = D.clf_metrics(yy, p, P, cls); preds[f"mlp:{r}"] = p
        if tname == "T2_side":
            p, P, cls = D.oof_classify(s["Cpre"][ii], yy, folds, "lr", seed)
            res["ctrl:prestim_counts"] = D.clf_metrics(yy, p, P, cls)
            yp = rng.permutation(yy)
            p, P, cls = D.oof_classify(R["counts"][ii], yp, folds, "lr", seed)
            res["ctrl:shuffled_labels"] = D.clf_metrics(yp, p, P, cls)
            st_ = s["still"][ii]
            if st_.sum() >= 60 and len(np.unique(yy[st_])) == 2:
                f2 = D.blocked_folds(int(st_.sum()))
                for r in ("counts", "umap", "stmap"):
                    p, P, cls = D.oof_classify(R[r][ii][st_], yy[st_], f2, "lr", seed)
                    res[f"ctrl:still_trials_{r}"] = D.clf_metrics(yy[st_], p, P, cls)
            err = (s["fb"][ii] < 0)
            if err.sum() >= 10:
                res["ctrl:error_trials_acc_counts"] = {"n": int(err.sum()), "accuracy": float(np.mean(preds["lr:counts"][err] == yy[err]))}
        comps = {"tcounts_vs_counts": ("lr:counts", "lr:tcounts"), "umap_vs_counts": ("lr:counts", "lr:umap"),
                 "umap_vs_counts_rp": ("lr:counts_rp", "lr:umap"), "umap_vs_umap_shuf": ("lr:umap_shuf", "lr:umap"),
                 "umap_vs_pmap": ("lr:pmap", "lr:umap"), "smap_vs_umap": ("lr:umap", "lr:smap"), "stmap_vs_tcounts": ("lr:tcounts", "lr:stmap"),
                 "countsumap_vs_countsumapshuf": ("lr:counts+umap_shuf", "lr:counts+umap"),
                 "mlp_stmap_vs_mlp_tcounts": ("mlp:tcounts", "mlp:stmap")}
        res["paired"] = {k: D.paired_boot(yy, preds[a], preds[b]) for k, (a, b) in comps.items() if a in preds and b in preds}
        out["tasks"][tname] = res
    # T4 continuous contralateral log contrast (ipsilateral-stimulus trials excluded)
    m = np.flatnonzero(s["ipsi"] == 0)
    y4 = np.log(s["contra"][m] + 0.03)
    folds = D.blocked_folds(len(m))
    out["tasks"]["T4_logcontrast"] = {r: D.reg_metrics(y4, D.oof_regress(R[r][m], y4, folds)) for r in ("counts", "tcounts", "umap", "smap", "stmap", "total")}
    return out


if __name__ == "__main__":
    split = sys.argv[1]
    inv = json.loads(Path("reports/phase13/ibl_inventory.json").read_text())
    t0 = time.time()
    res = {}
    for key in SPLITS[split]:
        if key not in inv or "status" in inv[key]:
            res[key] = {"status": "missing_dataset"}; continue
        try:
            res[key] = run_session(key)
        except Exception as e:  # noqa: BLE001
            res[key] = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
        print(key, res[key].get("status"), {t: {k: round(v["accuracy"], 3) for k, v in r.items() if isinstance(v, dict) and "accuracy" in v and k.startswith("lr:") and k in ("lr:counts", "lr:tcounts", "lr:umap", "lr:stmap", "lr:umap_shuf")}
                                             for t, r in res[key].get("tasks", {}).items() if isinstance(r, dict) and t != "T4_logcontrast"}, f"{time.time() - t0:.0f}s", flush=True)
    summary = {}
    ok = {k: v for k, v in res.items() if v.get("status") == "ok"}
    for t in ("T1_presence", "T2_side", "T3_contrast5", "T5_choice"):
        rows = [v["tasks"][t] for v in ok.values() if "paired" in v["tasks"].get(t, {})]
        if not rows:
            continue
        common = [r for r in rows[0] if all(r in x for x in rows) and isinstance(rows[0][r], dict)]
        summary[t] = {"n_sessions": len(rows),
                      "accuracy_mean": {r: float(np.mean([x[r]["accuracy"] for x in rows])) for r in common if "accuracy" in rows[0][r]},
                      "balanced_accuracy_mean": {r: float(np.mean([x[r]["balanced_accuracy"] for x in rows])) for r in common if "balanced_accuracy" in rows[0][r]},
                      "per_session_balanced_accuracy": {r: [x[r]["balanced_accuracy"] for x in rows] for r in ("lr:counts", "lr:tcounts", "lr:umap", "lr:stmap", "ctrl:prestim_counts", "ctrl:shuffled_labels") if r in common},
                      "majority_mean": float(np.mean([x["lr:counts"]["majority"] for x in rows])),
                      "paired_across_sessions": {k: D.across_sessions([x["paired"][k]["delta"] for x in rows if k in x["paired"]]) for k in rows[0]["paired"]},
                      "error_trials_accuracy": [x["ctrl:error_trials_acc_counts"]["accuracy"] for x in rows if "ctrl:error_trials_acc_counts" in x]}
    t4 = [v["tasks"]["T4_logcontrast"] for v in ok.values()]
    if t4:
        summary["T4_logcontrast"] = {r: {"r2_mean": float(np.mean([x[r]["r2"] for x in t4]))} for r in t4[0]}
    out = {"split": split, "protocol": __doc__, "sessions": res, "summary": summary, "runtime_s": time.time() - t0}
    Path(f"reports/phase13/ibl13_{split}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({t: {"bacc": {k: round(v, 3) for k, v in s_["balanced_accuracy_mean"].items() if not "noise" in k}, "maj": round(s_["majority_mean"], 3)}
                      for t, s_ in summary.items() if t != "T4_logcontrast"}, indent=0))
