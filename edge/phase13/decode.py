"""Phase 13 decoding core (runs in data/ibl-venv or data/allen-venv: needs scikit-learn only for the models).

Fixed protocol (decided before any evaluation; nothing is tuned on held-out sessions):
  folds        5 contiguous blocks of trials in time order (blocked CV: limits leakage from slow drifts and from
               correlated neighbouring trials); identical folds for every representation of a session
  models       L2 logistic regression (multinomial for > 2 classes) on standardized features, C chosen by an inner
               3-fold CV on the training folds only from {0.001, 0.01, 0.1, 1, 10}; a small MLP (one hidden layer of
               32 units, alpha 1e-2, early stopping) as the matched nonlinear model; ridge regression for continuous
               targets (alpha from {0.1, 1, 10, 100, 1000} by inner CV)
  metrics      accuracy, balanced accuracy, macro-F1, multiclass log loss, expected calibration error (10 bins),
               majority-class rate; R^2 and MAE for continuous targets
  statistics   per session paired trial bootstrap of accuracy differences (2000 resamples); across sessions mean
               difference, sign count, a session-level bootstrap interval and a Wilcoxon signed-rank p value
"""

from __future__ import annotations

import numpy as np

CS = (0.001, 0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def blocked_folds(n: int, k: int = 5) -> np.ndarray:
    return np.minimum(np.arange(n) * k // n, k - 1)


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def oof_classify(X: np.ndarray, y: np.ndarray, folds: np.ndarray, model: str = "lr", seed: int = 0):
    """Out-of-fold predicted labels and class probabilities [n, n_classes] (classes = np.unique(y))."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    classes = np.unique(y)
    P = np.zeros((len(y), len(classes)))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, Xte = _std(X[tr], X[te])
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:
            P[te] = np.mean(y[tr][:, None] == classes[None], axis=0); continue
        if model == "lr":
            n_min = np.min(np.unique(ytr, return_counts=True)[1])
            cv = StratifiedKFold(n_splits=max(2, min(3, n_min)), shuffle=True, random_state=seed)
            m = GridSearchCV(LogisticRegression(max_iter=3000), {"C": CS}, cv=cv, scoring="neg_log_loss").fit(Xtr, ytr)
        else:
            m = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, early_stopping=True, max_iter=500, random_state=seed).fit(Xtr, ytr)
        pr = m.predict_proba(Xte)
        cols = [int(np.flatnonzero(classes == c)[0]) for c in m.classes_]
        P[np.ix_(np.flatnonzero(te), cols)] = pr
    return classes[np.argmax(P, axis=1)], P, classes


def oof_regress(X: np.ndarray, y: np.ndarray, folds: np.ndarray):
    from sklearn.linear_model import RidgeCV
    pred = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, Xte = _std(X[tr], X[te])
        pred[te] = RidgeCV(alphas=ALPHAS).fit(Xtr, y[tr]).predict(Xte)
    return pred


def ece(P, y, classes, bins=10):
    conf = P.max(axis=1); pred = classes[np.argmax(P, axis=1)]; ok = pred == y
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def clf_metrics(y, pred, P, classes) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
    maj = float(np.max(np.unique(y, return_counts=True)[1]) / len(y))
    return {"n": int(len(y)), "n_classes": int(len(classes)), "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "log_loss": float(log_loss(y, np.clip(P, 1e-9, 1), labels=classes)), "ece": ece(P, y, classes), "majority": maj}


def reg_metrics(y, pred) -> dict:
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "r2": float(1 - np.sum((y - pred) ** 2) / max(ss, 1e-12)), "mae": float(np.mean(np.abs(y - pred))),
            "mae_mean_baseline": float(np.mean(np.abs(y - y.mean())))}


def paired_boot(y, pa, pb, n=2000, seed=0) -> dict:
    """Accuracy(b) - accuracy(a) with a trial-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    ca, cb = (pa == y).astype(float), (pb == y).astype(float)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = cb[idx].mean(axis=1) - ca[idx].mean(axis=1)
    return {"delta": float(cb.mean() - ca.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def across_sessions(deltas: list[float], n=5000, seed=0) -> dict:
    d = np.asarray([x for x in deltas if x is not None and np.isfinite(x)], float)
    if len(d) < 2:
        return {"n_sessions": int(len(d)), "mean": float(d.mean()) if len(d) else None}
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n, len(d))).mean(axis=1)
    from scipy.stats import wilcoxon
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = None
    return {"n_sessions": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "n_positive": int(np.sum(d > 0)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": p}


# ------------------------------------------------------------------------------------------------ representations
def bin_spikes(times: np.ndarray, clusters: np.ndarray, onsets: np.ndarray, units: np.ndarray, t0: float, t1: float,
               nbins: int, extra: np.ndarray | None = None):
    """Per trial: unit x time-bin counts [n_trials, n_units, nbins]; optionally the per-spike ``extra`` values and
    their unit / bin indices (for per-spike spatial maps)."""
    u_ix = {int(u): i for i, u in enumerate(units)}
    C = np.zeros((len(onsets), len(units), nbins), np.float32)
    per = []
    for t, on in enumerate(onsets):
        a, b = np.searchsorted(times, on + t0), np.searchsorted(times, on + t1)
        cl = clusters[a:b]; rel = times[a:b] - on
        bi = np.minimum(((rel - t0) / (t1 - t0) * nbins).astype(int), nbins - 1)
        ui = np.array([u_ix.get(int(c), -1) for c in cl], int)
        ok = ui >= 0
        np.add.at(C[t], (ui[ok], bi[ok]), 1)
        if extra is not None:
            per.append((ui[ok], bi[ok], extra[a:b][ok]))
    return C, per


def depth_map(pos: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(pos, edges, weights=weights)[0].astype(np.float32)


def unit_depth_maps(C: np.ndarray, unit_depth: np.ndarray, edges: np.ndarray, time_bins: bool = False) -> np.ndarray:
    """Spatial population map from per-unit counts placed at a fixed per-unit depth (static positions)."""
    B = np.stack([(unit_depth >= lo) & (unit_depth < hi) for lo, hi in zip(edges[:-1], edges[1:])], axis=0).astype(np.float32)  # [D, U]
    M = np.einsum("du,tub->tdb", B, C)                                                                                         # [T, D, nb]
    return M.reshape(len(C), -1) if time_bins else M.sum(axis=2)


def spike_depth_maps(per: list, n_trials: int, edges: np.ndarray, nbins: int, time_bins: bool, noise_um: float = 0.0, rng=None):
    D = len(edges) - 1
    out = np.zeros((n_trials, D, nbins if time_bins else 1), np.float32)
    for t, (ui, bi, dep) in enumerate(per):
        if len(dep) == 0:
            continue
        d = dep + (rng.normal(0, noise_um, len(dep)) if noise_um > 0 else 0.0)
        di = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, D - 1)
        inside = (d >= edges[0]) & (d < edges[-1])
        np.add.at(out[t], (di[inside], (bi[inside] if time_bins else np.zeros(inside.sum(), int))), 1)
    return out.reshape(n_trials, -1)


def random_projection(X: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    R = np.random.default_rng(seed).normal(size=(X.shape[1], dim)) / np.sqrt(dim)
    return X @ R
