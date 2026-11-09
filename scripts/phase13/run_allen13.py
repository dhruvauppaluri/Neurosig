"""Phase 13 Allen Brain Observatory (Visual Coding Neuropixels, DANDI 000021) decoding. Run in data/ibl-venv.
Usage: run_allen13.py within <split> | cross <train_split> <test_split>

Sessions (compact caches from fetch_allen13.py) are assigned to splits by sorted session id: the first 2 'dev', next
2 'select', the remaining 'final' (each session is a different mouse in this dataset; recorded in the report).
Window 0-250 ms after onset, 5 bins of 50 ms. Blocked 5-fold CV over presentation order. Blank presentations
(orientation / frame missing or -1) are excluded.
Tasks: DG orientation (8 classes), SG orientation (6), SG spatial frequency (5), NS natural-image identity (118);
continuous: DG orientation as (cos 2theta, sin 2theta) ridge regression -> median angular error.
Representations: counts, tcounts, total, area (per-area totals), dmap (unit counts in 40 um bins of probe vertical
position, per probe), dmap_shuf (positions permuted within probe), dmap_p12 (positions from the P12 line20 fit of the
unit mean waveform instead of the peak channel), counts_rp (dimension-matched random projection of counts).
Cross-session: a decoder trained on other mice and tested on a held-out mouse can only use features defined
identically across mice: VISp cortical-depth maps (vertical position relative to the shallowest VISp unit, 100 um
bins to 1000 um) x 5 time bins, versus VISp total count x 5 time bins and per-area totals x 5 time bins.
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from edge.phase13 import decode as D

CACHE = Path("data/phase13")
NB = 5


def sessions():
    s = sorted(p.stem.split("_", 1)[1] for p in CACHE.glob("allen_*.npz"))
    return {"dev": s[:2], "select": s[2:4], "final": s[4:]}


def load(sess):
    with np.load(CACHE / f"allen_{sess}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def p12_positions(d):
    from edge.phase11 import forward_models as F
    out = []
    for w, wp, vp, hp in zip(d["wave"], d["wave_pos"], d["unit_vpos"], d["unit_hpos"]):
        if not np.any(w):
            out.append(vp); continue
        w = w - w[:10].mean(axis=0)
        t0, pk = np.unravel_index(int(np.argmin(w)), w.shape)
        a = -w[t0]
        E = np.c_[wp, np.zeros(len(wp))]
        f = F.fit(F.Model("line20"), a, np.full(len(a), max(np.median(np.abs(w[:10])), 1e-3) + 0.05 * a.max()), E)
        out.append(float(f.theta[1]) if f.amps[0] > 0 else vp)
    return np.array(out, float)


def reps(d, key, pos_p12, rng):
    C = d[f"{key}_counts"].astype(np.float32)                                  # [T, U, 10] 25 ms bins
    C = C.reshape(C.shape[0], C.shape[1], NB, -1).sum(3)                        # 5 x 50 ms
    n = len(C)
    R = {"counts": C.sum(2), "tcounts": C.reshape(n, -1), "total": C.sum((1, 2))[:, None]}
    areas = sorted(set(d["unit_area"].tolist()))
    R["area"] = np.stack([C[:, d["unit_area"] == a].sum((1, 2)) for a in areas], axis=1)
    maps, maps_s, maps_p = [], [], []
    for p in np.unique(d["unit_probe"]):
        m = d["unit_probe"] == p
        v = d["unit_vpos"][m]; edges = np.arange(np.floor(v.min() / 40) * 40, np.ceil(v.max() / 40 + 1) * 40 + 1, 40.0)
        Cm = C[:, m]
        maps.append(D.unit_depth_maps(Cm, v, edges)); maps_s.append(D.unit_depth_maps(Cm, rng.permutation(v), edges))
        maps_p.append(D.unit_depth_maps(Cm, np.clip(pos_p12[m], edges[0], edges[-1] - 1e-6), edges))
    R["dmap"], R["dmap_shuf"], R["dmap_p12"] = np.hstack(maps), np.hstack(maps_s), np.hstack(maps_p)
    R["counts_rp"] = D.random_projection(R["counts"], R["dmap"].shape[1], seed=1)
    return R


def labels(d):
    T = {}
    o = d["dg_orientation"]; T["DG_orientation8"] = ("dg", np.isfinite(o), np.nan_to_num(o).round().astype(int))
    o = d["sg_orientation"]; T["SG_orientation6"] = ("sg", np.isfinite(o), np.nan_to_num(o).round().astype(int))
    sf = d["sg_spatial_frequency"]; T["SG_sf5"] = ("sg", np.isfinite(sf), np.round(np.nan_to_num(sf) * 100).astype(int))
    fr = d["ns_frame"]; T["NS_image118"] = ("ns", np.isfinite(fr) & (fr >= 0), np.nan_to_num(fr, nan=-1).astype(int))
    return T


def within(split):
    out = {}
    for sess in sessions()[split]:
        d = load(sess); rng = np.random.default_rng(20271601)
        pos = p12_positions(d)
        o = {"n_units": int(len(d["unit_id"])), "areas": sorted(set(d["unit_area"].tolist())),
             "p12_vs_peak_vpos_shift_um_median": float(np.median(np.abs(pos - d["unit_vpos"]))), "tasks": {}}
        cache = {}
        for t, (key, m, y) in labels(d).items():
            if key not in cache:
                cache[key] = reps(d, key, pos, rng)
            R = cache[key]
            ii = np.flatnonzero(m); yy = y[ii]; folds = D.blocked_folds(len(ii))
            use = ("counts", "tcounts", "total", "area", "dmap", "dmap_shuf", "dmap_p12", "counts_rp") if t != "NS_image118" \
                else ("counts", "tcounts", "total", "area", "dmap", "dmap_shuf")
            res, preds = {}, {}
            for r in use:
                p, P, cls = D.oof_classify(R[r][ii], yy, folds, "lr")
                res[f"lr:{r}"] = D.clf_metrics(yy, p, P, cls); preds[r] = p
            if t != "NS_image118":
                yp = rng.permutation(yy); p, P, cls = D.oof_classify(R["counts"][ii], yp, folds, "lr")
                res["ctrl:shuffled_labels"] = D.clf_metrics(yp, p, P, cls)
            comps = {"tcounts_vs_counts": ("counts", "tcounts"), "dmap_vs_counts": ("counts", "dmap"), "dmap_vs_counts_rp": ("counts_rp", "dmap"),
                     "dmap_vs_dmap_shuf": ("dmap_shuf", "dmap"), "dmap_p12_vs_dmap": ("dmap", "dmap_p12"), "area_vs_total": ("total", "area")}
            res["paired"] = {k: D.paired_boot(yy, preds[a], preds[b]) for k, (a, b) in comps.items() if a in preds and b in preds}
            o["tasks"][t] = res
            print(sess, t, {k: round(v["accuracy"], 3) for k, v in res.items() if isinstance(v, dict) and "accuracy" in v}, flush=True)
        # continuous orientation (DG): ridge on (cos 2 theta, sin 2 theta)
        key, m, y = labels(d)["DG_orientation8"]; ii = np.flatnonzero(m); th = np.deg2rad(y[ii]); folds = D.blocked_folds(len(ii))
        o["tasks"]["DG_orientation_continuous"] = {}
        for r in ("counts", "tcounts", "dmap"):
            pc = D.oof_regress(cache["dg"][r][ii], np.cos(2 * th), folds); ps = D.oof_regress(cache["dg"][r][ii], np.sin(2 * th), folds)
            err = np.abs(np.angle(np.exp(1j * (np.arctan2(ps, pc) - 2 * th)))) / 2
            o["tasks"]["DG_orientation_continuous"][r] = {"median_abs_error_deg": float(np.rad2deg(np.median(err))), "chance_deg": 45.0}
        out[sess] = o
    return out


def cortical_maps(d, key):
    C = d[f"{key}_counts"].astype(np.float32); C = C.reshape(C.shape[0], C.shape[1], NB, -1).sum(3)
    v = d["unit_vpos"]; m = d["unit_area"] == "VISp"
    if m.sum() < 5:
        return None
    top = v[m].max()                                         # shallowest VISp unit (largest vertical position = closest to surface)
    depth = top - v[m]
    edges = np.arange(0, 1001, 100.0)
    M = D.unit_depth_maps(C[:, m], np.clip(depth, 0, 999.9), edges, time_bins=True)
    tot = C[:, m].sum(1)
    areas = ("VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm")
    A = np.stack([C[:, d["unit_area"] == a].sum(1) if np.any(d["unit_area"] == a) else np.zeros((len(C), NB)) for a in areas], axis=1).reshape(len(C), -1)
    return {"vmap": M, "vtotal": tot, "areas": A}


def cross(train_split, test_split):
    S = sessions(); out = {"train": S[train_split] if train_split != "dev+select" else S["dev"] + S["select"], "test": S[test_split], "tasks": {}}
    train = out["train"]
    for t in ("DG_orientation8", "NS_image118", "SG_orientation6"):
        key = labels(load(train[0]))[t][0]
        def feats(sess):
            d = load(sess); key_, m, y = labels(d)[t]; f = cortical_maps(d, key_)
            return None if f is None else ({k: v[m] for k, v in f.items()}, y[m])
        tr = [feats(s) for s in train]; tr = [x for x in tr if x is not None]
        res = {}
        for test in S[test_split]:
            te = feats(test)
            if te is None:
                continue
            r = {}
            for rep in ("vmap", "vtotal", "areas"):
                from sklearn.linear_model import LogisticRegression
                Xtr = np.vstack([x[0][rep] for x in tr]); ytr = np.concatenate([x[1] for x in tr])
                # per-session z-scoring removes overall rate / unit-number differences between mice
                z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)
                Xtr = np.vstack([z(x[0][rep]) for x in tr])
                m = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr, ytr)
                p = m.predict(z(te[0][rep]))
                r[rep] = {"accuracy": float(np.mean(p == te[1])), "chance": 1.0 / len(np.unique(te[1]))}
            res[test] = r
        out["tasks"][t] = res
        print(t, res, flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "within":
        split = sys.argv[2]
        res = {"protocol": __doc__, "split": split, "sessions_by_split": sessions(), "sessions": within(split)}
        name = f"allen13_within_{split}"
    else:
        res = {"protocol": __doc__, **cross(sys.argv[2], sys.argv[3])}
        name = f"allen13_cross_{sys.argv[2]}_to_{sys.argv[3]}"
    res["runtime_s"] = time.time() - t0
    Path(f"reports/phase13/{name}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
