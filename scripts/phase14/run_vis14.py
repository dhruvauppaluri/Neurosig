"""Phase 14V: visual decoding on Allen Visual Coding Neuropixels with the corrected waveform map (data/ibl-venv).

Usage:
  run_vis14.py within <split>            split in dev | select | final  (final only after frozen_p14.json exists)
  run_vis14.py cross <split>             leave-one-mouse-out within select; final = train on dev+select, test each
                                         final mouse once
Splits (reports/phase14/allen_split14.json, reserved before fetching): dev = 2 Phase 13 dev mice; select = 3 Phase 13
select/final mice + the first 6 eligible 'train expansion' mice; final = the first 6 eligible 'final candidate'
mice. Every session is a different mouse.

Protocol (fixed on dev before select/final): window 0-250 ms after onset (5 ms bins in the cache), blank
presentations excluded, blocked 5-fold CV in presentation order, L2 logistic regression with inner-CV C, a matched
one-hidden-layer MLP (32 units) for the temporal comparison, paired trial bootstrap within session and session-level
Wilcoxon / bootstrap across sessions.
Tasks: DG orientation (8), SG orientation (6), SG spatial frequency (5), NS image identity (118), Gabor x and y
position (receptive-field mapping stimulus, where cached).
Representations:
  temporal (Section 9)  counts | tb50 (5 bins) | tb25 (10 bins) | lat (count + spike-time centroid per unit) |
                        basis (6 raised-cosine temporal bases per unit) | area_tb{50,25,10,5} (per-area totals)
  spatial (Section 11)  dmap_peak (peak-channel depth, 40 um bins per probe) | dmap_p12c (P12 line20 position of the
                        CORRECTED mean waveform) | dmap_shuf (positions permuted within probe) | dmap_rand (uniform
                        random coordinates) | counts_rp (dimension-matched random projection) | ccf_surface (CCF
                        AP x ML 200 um grid totals: cortical-surface position, the retinotopically organized axis)
Cross-mouse (Section 10): per-area x time features (area_tb25) with no normalization | per-session z-score (label-free)
| per-session z-score + permuted area identities (control) | shuffled training labels (control).
Parameterized-stimulus reconstruction (Section 12): SG (orientation, SF) combinations split so that test combinations
are never seen in training; ridge regression of (cos 2theta, sin 2theta) and log SF.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from edge.phase13 import decode as D

CACHE = Path("data/phase14")
SPLIT = json.loads(Path("reports/phase14/allen_split14.json").read_text())


def sessions():
    inv = json.loads(Path("reports/phase14/allen_inventory14.json").read_text())
    ok = lambda s: inv.get(s, {}).get("status") == "ok"
    tr = [s for s in SPLIT["train_candidates"] if ok(s)][:6]; fi = [s for s in SPLIT["final_candidates"] if ok(s)][:6]
    return {"dev": [s for s in SPLIT["dev"] if ok(s)], "select": [s for s in SPLIT["select_p13"] if ok(s)] + tr, "final": fi}


def load(sess):
    with np.load(CACHE / f"allen_{sess}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    g = CACHE / f"allen_gabor_{sess}.npz"
    if g.exists():
        with np.load(g) as z:
            d.update({f"gb_{k}": z[k] for k in z.files})
    return d


def p12c_positions(d):
    """Vertical position of each unit from the line20 fit of its corrected mean waveform (trough amplitudes)."""
    from edge.phase11 import forward_models as F
    out = []
    for w, wp, vp in zip(d["wave"], d["wave_pos"], d["unit_vpos"]):
        if not np.any(w):
            out.append(vp); continue
        w = w - w[:10].mean(axis=0)
        t0, _ = np.unravel_index(int(np.argmin(w)), w.shape)
        a = -w[t0]
        f = F.fit(F.Model("line20"), a, np.full(len(a), max(np.median(np.abs(w[:10])), 1e-9) + 0.05 * max(a.max(), 1e-9)), np.c_[wp, np.zeros(len(wp))])
        out.append(float(f.theta[1]) if f.amps[0] > 0 else vp)
    return np.array(out, float)


def raised_cosine(nb=6, n=50):
    t = np.log1p(np.arange(n)); c = np.linspace(t[0], t[-1], nb); w = (c[1] - c[0]) * 1.5
    B = np.array([np.where(np.abs(t - ci) < w, 0.5 * (1 + np.cos(np.pi * (t - ci) / w)), 0.0) for ci in c])
    return B / B.sum(axis=1, keepdims=True)


def temporal_reps(C, area):
    n, U, T = C.shape
    R = {"counts": C.sum(2)}
    for w in (50, 25):
        R[f"tb{w}"] = C.reshape(n, U, T // (w // 5), -1).sum(3).reshape(n, -1)
    tt = (np.arange(T) + 0.5) * 5.0
    cnt = C.sum(2); cen = np.where(cnt > 0, (C * tt).sum(2) / np.maximum(cnt, 1), 125.0)
    R["lat"] = np.hstack([cnt, cen])
    R["basis"] = np.einsum("nut,bt->nub", C, raised_cosine()).reshape(n, -1)
    areas = sorted(set(area.tolist()))
    A = np.stack([C[:, area == a].sum(1) for a in areas], axis=1)              # [n, areas, 50]
    for w in (50, 25, 10, 5):
        R[f"area_tb{w}"] = A.reshape(n, len(areas), T // (w // 5), -1).sum(3).reshape(n, -1)
    return R


def spatial_reps(C, d, pos_p12, rng):
    cnt = C.sum(2); R = {}
    maps = {k: [] for k in ("dmap_peak", "dmap_p12c", "dmap_shuf", "dmap_rand")}
    for p in np.unique(d["unit_probe"]):
        m = d["unit_probe"] == p
        v = d["unit_vpos"][m]; edges = np.arange(np.floor(v.min() / 40) * 40, np.ceil(v.max() / 40 + 1) * 40 + 1, 40.0)
        cm = C[:, m]
        maps["dmap_peak"].append(D.unit_depth_maps(cm, v, edges))
        maps["dmap_p12c"].append(D.unit_depth_maps(cm, np.clip(pos_p12[m], edges[0], edges[-1] - 1e-6), edges))
        maps["dmap_shuf"].append(D.unit_depth_maps(cm, rng.permutation(v), edges))
        maps["dmap_rand"].append(D.unit_depth_maps(cm, rng.uniform(edges[0], edges[-1] - 1e-6, len(v)), edges))
    for k, v in maps.items():
        R[k] = np.hstack(v)
    R["counts_rp"] = D.random_projection(cnt, R["dmap_peak"].shape[1], seed=1)
    ccf = d["unit_ccf"]
    ok = np.all(np.isfinite(ccf), axis=1) & (ccf[:, 0] > 0)
    if ok.sum() >= 10:
        ap = np.floor(ccf[:, 0] / 200).astype(int); ml = np.floor(ccf[:, 2] / 200).astype(int)
        keys = sorted(set(zip(ap[ok].tolist(), ml[ok].tolist())))
        R["ccf_surface"] = np.stack([cnt[:, ok & (ap == a) & (ml == b)].sum(1) for a, b in keys], axis=1)
    return R


def labels(d):
    T = {}
    o = d["dg_orientation"]; T["DG_orientation8"] = ("dg", np.isfinite(o), np.nan_to_num(o).round().astype(int))
    o = d["sg_orientation"]; T["SG_orientation6"] = ("sg", np.isfinite(o), np.nan_to_num(o).round().astype(int))
    sf = d["sg_spatial_frequency"]; T["SG_sf5"] = ("sg", np.isfinite(sf), np.round(np.nan_to_num(sf) * 100).astype(int))
    fr = d["ns_frame"]; T["NS_image118"] = ("ns", np.isfinite(fr) & (fr >= 0), np.nan_to_num(fr, nan=-1).astype(int))
    if "gb_x" in d:
        T["GB_x"] = ("gb", np.isfinite(d["gb_x"]), np.nan_to_num(d["gb_x"]).round().astype(int))
        T["GB_y"] = ("gb", np.isfinite(d["gb_y"]), np.nan_to_num(d["gb_y"]).round().astype(int))
    return T


def counts_of(d, key):
    if key == "gb":
        C = d["gb_counts"].astype(np.float32)                                    # 25 ms bins -> pseudo 5 ms grid
        return np.repeat(C, 5, axis=2) / 5.0
    return d[f"{key}_counts"].astype(np.float32)


TEMPORAL = ("counts", "tb50", "tb25", "lat", "basis", "area_tb50", "area_tb25", "area_tb10", "area_tb5")
SPATIAL = ("dmap_peak", "dmap_p12c", "dmap_shuf", "dmap_rand", "counts_rp", "ccf_surface")
COMPS = {"tb25_vs_counts": ("counts", "tb25"), "tb50_vs_counts": ("counts", "tb50"), "lat_vs_counts": ("counts", "lat"),
         "basis_vs_counts": ("counts", "basis"), "area_tb5_vs_area_tb50": ("area_tb50", "area_tb5"),
         "area_tb10_vs_area_tb50": ("area_tb50", "area_tb10"), "area_tb25_vs_area_tb50": ("area_tb50", "area_tb25"),
         "dmap_peak_vs_counts": ("counts", "dmap_peak"), "dmap_p12c_vs_dmap_peak": ("dmap_peak", "dmap_p12c"),
         "dmap_p12c_vs_dmap_shuf": ("dmap_shuf", "dmap_p12c"), "dmap_p12c_vs_dmap_rand": ("dmap_rand", "dmap_p12c"),
         "dmap_p12c_vs_counts_rp": ("counts_rp", "dmap_p12c"), "ccf_surface_vs_counts": ("counts", "ccf_surface"),
         "mlp_tb25_vs_mlp_counts": ("mlp:counts", "mlp:tb25")}


def within_session(sess):
    d = load(sess); rng = np.random.default_rng(20271451)
    pos = p12c_positions(d)
    o = {"n_units": int(len(d["unit_id"])), "areas": sorted(set(d["unit_area"].tolist())),
         "trough_on_peak_frac": float(np.nanmean(d["unit_trough_peak_dist"] == 0)),
         "p12c_vs_peak_shift_um_median": float(np.median(np.abs(pos - d["unit_vpos"]))), "tasks": {}}
    cache = {}
    for t, (key, m, y) in labels(d).items():
        if key not in cache:
            C = counts_of(d, key)
            cache[key] = {**temporal_reps(C, d["unit_area"]), **spatial_reps(C, d, pos, rng)}
        R = cache[key]
        ii = np.flatnonzero(m); yy = y[ii]; folds = D.blocked_folds(len(ii))
        res, preds = {}, {}
        use = list(TEMPORAL) + [r for r in SPATIAL if r in R]
        if t == "NS_image118":
            use = [r for r in use if r not in ("basis", "lat")]
        for r in use:
            p, P, cls = D.oof_classify(R[r][ii], yy, folds, "lr")
            res[f"lr:{r}"] = D.clf_metrics(yy, p, P, cls); preds[r] = p
        if t != "NS_image118":
            for r in ("counts", "tb25"):
                p, P, cls = D.oof_classify(R[r][ii], yy, folds, "mlp")
                res[f"mlp:{r}"] = D.clf_metrics(yy, p, P, cls); preds[f"mlp:{r}"] = p
            yp = rng.permutation(yy); p, P, cls = D.oof_classify(R["counts"][ii], yp, folds, "lr")
            res["ctrl:shuffled_labels"] = D.clf_metrics(yp, p, P, cls)
        res["paired"] = {k: D.paired_boot(yy, preds[a], preds[b]) for k, (a, b) in COMPS.items() if a in preds and b in preds}
        o["tasks"][t] = res
        print(sess, t, {k: round(v["balanced_accuracy"], 3) for k, v in res.items() if isinstance(v, dict) and "balanced_accuracy" in v}, flush=True)
    # Section 12: unseen (orientation, SF) combinations
    key, m, ori = labels(d)["SG_orientation6"]; _, m2, sf = labels(d)["SG_sf5"]
    ii = np.flatnonzero(m & m2)
    combos = sorted(set(zip(ori[ii].tolist(), sf[ii].tolist())))
    rs = np.random.default_rng(7); test_c = set(map(tuple, np.array(combos)[rs.choice(len(combos), len(combos) // 3, replace=False)].tolist()))
    te = np.array([(a, b) in test_c for a, b in zip(ori[ii], sf[ii])])
    R = cache["sg"]; o["unseen_combo"] = {}
    from sklearn.linear_model import RidgeCV
    for r in ("counts", "tb25", "area_tb25"):
        X = R[r][ii]; mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9; Z = (X - mu) / sd
        th = np.deg2rad(ori[ii])
        pc = RidgeCV(alphas=D.ALPHAS).fit(Z[~te], np.cos(2 * th[~te])).predict(Z[te])
        ps = RidgeCV(alphas=D.ALPHAS).fit(Z[~te], np.sin(2 * th[~te])).predict(Z[te])
        err = np.abs(np.angle(np.exp(1j * (np.arctan2(ps, pc) - 2 * th[te])))) / 2
        lsf = np.log(sf[ii] / 100.0)
        psf = RidgeCV(alphas=D.ALPHAS).fit(Z[~te], lsf[~te]).predict(Z[te])
        o["unseen_combo"][r] = {"orientation_median_err_deg": float(np.rad2deg(np.median(err))), "orientation_chance_deg": 45.0,
                                "logsf_r2": D.reg_metrics(lsf[te], psf)["r2"], "n_test": int(te.sum()), "n_test_combos": len(test_c)}
    return sess, o


def within(split):
    from multiprocessing import Pool
    S = sessions()[split]
    with Pool(min(4, len(S))) as p:
        return dict(p.map(within_session, S))


def summarize(out):
    summ = {}
    tasks = sorted({t for o in out.values() for t in o["tasks"]})
    for t in tasks:
        S = [o["tasks"][t] for o in out.values() if t in o["tasks"]]
        summ[t] = {"balanced_accuracy_mean": {k: float(np.mean([s[k]["balanced_accuracy"] for s in S if k in s]))
                                              for k in S[0] if k != "paired"},
                   "paired_across_sessions": {k: D.across_sessions([s["paired"][k]["delta"] for s in S if k in s["paired"]])
                                              for k in S[0]["paired"]}}
    return summ


# ------------------------------------------------------------------------------------------------ cross-mouse
def cross_feats(sess, t):
    d = load(sess)
    lab = labels(d)
    if t not in lab:
        return None
    key, m, y = lab[t]
    C = counts_of(d, key)
    areas = ("VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm")
    n, U, T = C.shape
    A = np.stack([C[:, d["unit_area"] == a].sum(1) if np.any(d["unit_area"] == a) else np.zeros((n, T)) for a in areas], axis=1)
    A = A.reshape(n, len(areas), 10, -1).sum(3)                                 # 25 ms
    return A[m].reshape(int(m.sum()), -1), y[m], A[m]


def cross(split):
    from sklearn.linear_model import LogisticRegression
    S = sessions()
    if split == "select":
        pairs = [([s for s in S["dev"] + S["select"] if s != te], te) for te in S["select"]]
    else:
        pairs = [(S["dev"] + S["select"], te) for te in S["final"]]
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = {}
    for t in ("DG_orientation8", "SG_orientation6", "SG_sf5", "NS_image118", "GB_x", "GB_y"):
        res = {}
        for tr, te in pairs:
            ftr = [cross_feats(s, t) for s in tr]; ftr = [f for f in ftr if f is not None]; fte = cross_feats(te, t)
            if fte is None or not ftr:
                continue
            ytr = np.concatenate([f[1] for f in ftr]); r = {"chance": 1.0 / len(np.unique(fte[1])), "n_train_mice": len(ftr)}
            variants = {"raw": (np.vstack([f[0] for f in ftr]), fte[0]),
                        "zscore": (np.vstack([z(f[0]) for f in ftr]), z(fte[0]))}
            perm = np.random.default_rng(3).permutation(6)
            variants["zscore_area_perm"] = (variants["zscore"][0], z(fte[2][:, perm].reshape(len(fte[2]), -1)))
            for name, (Xtr, Xte) in variants.items():
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                mdl = LogisticRegression(C=0.1, max_iter=3000).fit((Xtr - mu) / sd, ytr)
                p = mdl.predict((Xte - mu) / sd)
                from sklearn.metrics import balanced_accuracy_score
                r[name] = float(balanced_accuracy_score(fte[1], p))
            ysh = np.random.default_rng(4).permutation(ytr)
            Xtr, Xte = variants["zscore"]; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            from sklearn.metrics import balanced_accuracy_score
            r["shuffled_train_labels"] = float(balanced_accuracy_score(fte[1], LogisticRegression(C=0.1, max_iter=3000).fit((Xtr - mu) / sd, ysh).predict((Xte - mu) / sd)))
            res[te] = r
            print(t, te, r, flush=True)
        out[t] = res
    return out


if __name__ == "__main__":
    mode, split = sys.argv[1], sys.argv[2]
    if split == "final" and not Path("reports/phase14/frozen_p14.json").exists():
        raise SystemExit("final evaluation requires reports/phase14/frozen_p14.json")
    t0 = time.time()
    if mode == "within":
        out = within(split)
        res = {"protocol": __doc__, "split": split, "sessions_by_split": sessions(), "sessions": out, "summary": summarize(out)}
    else:
        res = {"protocol": __doc__, "split": split, "sessions_by_split": sessions(), "tasks": cross(split)}
    res["runtime_s"] = time.time() - t0
    Path(f"reports/phase14/vis14_{mode}_{split}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
