"""Phase 15 Track D: cross-animal visual decoding from label-free, anatomically / functionally grouped populations.

No unit identity is shared across mice, so every feature is defined on groups that exist in every mouse. A unit's
group uses only label-free information:
  area     Allen CCF structure of the unit's peak channel (VISp, VISl, VISal, VISrl, VISam, VISpm)
  layer    relative depth within the probe's visual-cortex span (superficial / middle / deep thirds)
  type     waveform trough-to-peak duration (narrow < 0.40 ms vs broad), a fast-spiking / regular-spiking proxy
  fclust   functional cluster of the unit's stimulus-AVERAGED response time course (mean over all presentations of a
           stimulus block, labels never used); k-means centres fitted on training mice only, test units assigned to the
           nearest centre
Feature = per group x time bin, the mean over the group's units of each unit's normalized response.
Unit normalization (both label-free):
  rate     inductive: divide by the unit's whole-session firing rate (NWB metadata, not the evaluation trials)
  z        transductive, label-free: z-score each unit over the evaluation block's trials (labels unused)
Session-level feature normalization: 'z' (per-session z-score of each feature over the block's trials; label-free
transductive) or 'none'. Optional label-free transductive CORAL alignment of the held-out mouse's feature covariance
to the training covariance. Evaluation: leave-one-mouse-out; nothing from the held-out mouse except its unlabeled
responses (transductive variants only) enters training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AREAS = ("VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm")


@dataclass
class Spec:
    groups: tuple = ("area",)          # subset of area, layer, type, fclust
    bin_ms: int = 25
    unit_norm: str = "rate"            # rate | z | none
    sess_norm: str = "z"               # z | none
    coral: bool = False
    k_fclust: int = 6
    t_max_ms: int = 250
    model: str = "lr"                  # lr | ridge | mlp
    weight: str = "none"               # none | snr | quality: unit-quality-weighted pooling (Track E, label-free)
    C: float = 0.1
    extra: dict = field(default_factory=dict)


def unit_groups(d: dict) -> dict:
    """Label-free per-unit attributes: area index, layer third, narrow/broad type."""
    area = np.array([AREAS.index(a) if a in AREAS else -1 for a in d["unit_area"]])
    layer = np.zeros(len(area), int)
    for p in np.unique(d["unit_probe"]):
        m = (d["unit_probe"] == p) & (area >= 0)
        if m.sum() < 3:
            continue
        v = d["unit_vpos"][m]; top, bot = v.max(), v.min()
        rel = (top - v) / max(top - bot, 1e-6)
        layer[np.flatnonzero(m)] = np.minimum((rel * 3).astype(int), 2)
    W = d["wave"]                                        # [U, 82, 32] mean waveform (corrected map)
    typ = np.zeros(len(area), int)
    for i, w in enumerate(W):
        if not np.any(w):
            continue
        tr = np.unravel_index(int(np.argmin(w)), w.shape)
        x = w[:, tr[1]]; t0 = tr[0]
        pk = t0 + int(np.argmax(x[t0:])) if t0 < len(x) - 1 else t0
        typ[i] = int((pk - t0) / 30.0 < 0.40)            # 1 = narrow (30 kHz)
    return {"area": area, "layer": layer, "type": typ}


def unit_responses(C: np.ndarray, d: dict, spec: Spec) -> np.ndarray:
    """[trials, units, bins] normalized responses at spec.bin_ms over 0..t_max_ms."""
    nb = spec.t_max_ms // 5
    X = C[:, :, :nb].astype(np.float32)
    k = spec.bin_ms // 5
    X = X.reshape(X.shape[0], X.shape[1], nb // k, k).sum(3)
    if spec.unit_norm == "rate":
        r = np.maximum(np.asarray(d["unit_firing_rate"] if "unit_firing_rate" in d else d["unit_rate"], float), 0.1)
        X = X / (r[None, :, None] * spec.bin_ms / 1000.0)
    elif spec.unit_norm == "z":
        m = X.mean(axis=(0, 2), keepdims=True); s = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X = (X - m) / s
    return X


def fclust_profiles(C: np.ndarray, spec: Spec) -> np.ndarray:
    """Label-free unit descriptor: stimulus-averaged PSTH (10 ms) normalized to unit peak."""
    nb = spec.t_max_ms // 5
    P = C[:, :, :nb].astype(np.float32).mean(axis=0).reshape(C.shape[1], nb // 2, 2).sum(2)
    P = P - P[:, :2].mean(axis=1, keepdims=True)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-6)


def kmeans(X, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmin(((X[:, None] - c[None]) ** 2).sum(2), axis=1)
        for j in range(k):
            if np.any(a == j):
                c[j] = X[a == j].mean(0)
    return c


def features(C, d, spec: Spec, centres=None) -> np.ndarray:
    X = unit_responses(C, d, spec)
    g = unit_groups(d)
    keys = [g["area"]]
    sizes = [len(AREAS)]
    if "layer" in spec.groups:
        keys.append(g["layer"]); sizes.append(3)
    if "type" in spec.groups:
        keys.append(g["type"]); sizes.append(2)
    if "fclust" in spec.groups and centres is not None:
        P = fclust_profiles(C, spec)
        keys.append(np.argmin(((P[:, None] - centres[None]) ** 2).sum(2), axis=1)); sizes.append(len(centres))
    if "area" not in spec.groups:
        keys = keys[1:]; sizes = sizes[1:]
    valid = g["area"] >= 0
    gid = np.zeros(len(valid), int)
    for k, s in zip(keys, sizes):
        gid = gid * s + k
    G = int(np.prod(sizes))
    F = np.zeros((X.shape[0], G, X.shape[2]), np.float32)
    if spec.weight == "snr":
        w = np.clip(np.nan_to_num(np.asarray(d["unit_snr"], float)), 0.1, 10.0)
    elif spec.weight == "quality":                   # good isolation and few refractory violations weigh more
        iso = np.nan_to_num(np.asarray(d["unit_isolation_distance"], float), nan=20.0)
        isi = np.nan_to_num(np.asarray(d["unit_isi_violations"], float), nan=1.0)
        w = np.clip(iso / 50.0, 0.1, 2.0) / (1.0 + 10.0 * np.clip(isi, 0, 1))
    else:
        w = np.ones(len(valid))
    for j in range(G):
        m = valid & (gid == j)
        if m.any():
            F[:, j] = np.tensordot(X[:, m], w[m] / w[m].sum(), axes=([1], [0]))
    F = F.reshape(X.shape[0], -1)
    if spec.sess_norm == "z":
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    return F


def coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Map target features so their covariance matches the source covariance (Sun & Saenko 2016); label-free."""
    def msqrt(A, inv=False):
        w, V = np.linalg.eigh(A); w = np.maximum(w, 1e-8)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1]); Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    mu_t = Xt.mean(0)
    return (Xt - mu_t) @ msqrt(Ct, inv=True) @ msqrt(Cs) + Xs.mean(0)


def fit_predict(Xtr, ytr, Xte, spec: Spec, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    if spec.model == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=spec.C, max_iter=3000).fit(A, ytr)
        return m.predict(B), m.predict_proba(B), m.classes_
    if spec.model == "ridge":
        from sklearn.linear_model import RidgeClassifier
        m = RidgeClassifier(alpha=1.0 / spec.C).fit(A, ytr)
        p = m.predict(B); cls = m.classes_
        return p, np.eye(len(cls))[np.searchsorted(cls, p)], cls
    if spec.model == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(spec.extra.get("hidden", 64),), alpha=spec.extra.get("alpha", 1.0), early_stopping=True,
                          max_iter=400, random_state=seed).fit(A, ytr)
        return m.predict(B), m.predict_proba(B), m.classes_
    if spec.model == "ens":
        from sklearn.linear_model import LogisticRegression
        Ps = []
        for c in (0.01, 0.1, 1.0):
            m = LogisticRegression(C=c, max_iter=3000).fit(A, ytr); Ps.append(m.predict_proba(B))
        P = np.mean(Ps, axis=0); cls = m.classes_
        return cls[np.argmax(P, 1)], P, cls
    raise ValueError(spec.model)
