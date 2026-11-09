"""Phase 13 / Section 11 controlled simulation: when does localization quality change stimulus decoding?

Fixed design (no real data; known positions). Each simulated session has 60 new neurons at uniform depths 0-1000 um
(identities never shared across sessions). K = 4 stimuli, 40 trials each, Poisson counts in a 100 ms window,
baseline 5 Hz, peak drive 40 Hz, log-normal gains (sigma 0.5).
  M1 spatially organized   stimulus k drives neurons near depth c_k (150 / 400 / 650 / 900 um, Gaussian width 80 um)
  M2 salt-and-pepper       each neuron's preferred stimulus is random, independent of depth (same rate statistics)
Decoders (L2 logistic, C = 0.1, features z-scored within session):
  within-session counts    unit-identity features, trained and tested inside a session (blocked 5-fold): position-free
  cross-session depth map  20 bins of 50 um built from ESTIMATED positions; train on 7 sessions, test on the 8th
Position error: unit-level static error N(0, sigma_u) per neuron, sigma_u in {0, 3, 5, 12, 25, 50, 100, 200} um,
and permuted positions; event-level error N(0, sigma_e) per spike (trial-varying), sigma_e in {0, 12, 25, 50, 100} um.
Reference precisions from Phase 12 (real data, proxies): unit-level pooled l20_robust (N = 25-50 spikes) ~2-4 um
pairwise disagreement; single-event median deviation ~10-14 um. 10 repetitions with different seeds.

Revision after a pilot (recorded, not hidden): with 250 um bands and 40 Hz drive every decoder saturated (100% up to
sigma 100 um), which cannot show where precision matters. The reported design sweeps the spatial scale instead:
M1 becomes a periodic code (stimulus k preferred in bands of width delta repeating every K*delta um, Gaussian tuning
of width delta/3, peak drive 15 Hz), delta in {20, 50, 100, 250} um, depth-map bins of delta/2; 5 seeds per cell.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")

K, N, TR, S = 4, 60, 40, 8
CENT = np.array([150.0, 400.0, 650.0, 900.0]); WIDTH = 80.0
EDGES = np.arange(0, 1001, 50.0)


DELTA = 250.0


def make_session(rng, model):
    depth = rng.uniform(0, 1000, N); gain = rng.lognormal(0, 0.5, N)
    if model == "M1":
        period = K * DELTA
        cen = (np.arange(K)[:, None] + 0.5) * DELTA                                       # band centres within a period
        dd = np.abs(((depth[None] - cen) + period / 2) % period - period / 2)              # circular distance [K, N]
        drive = np.exp(-dd ** 2 / (2 * (DELTA / 3) ** 2))
    else:
        pref = rng.integers(0, K, N); drive = np.zeros((K, N)); drive[pref, np.arange(N)] = 1.0
        drive = drive * np.exp(-0.0 * depth[None])
        # match the M1 per-neuron mean drive so only the spatial organization differs
        drive = drive * 0.33
    rate = 5.0 + 15.0 * drive * gain[None]                                                # Hz [K, N]
    y = np.repeat(np.arange(K), TR); rng.shuffle(y)
    counts = rng.poisson(rate[y] * 0.1)                                                    # [T, N]
    return depth, counts, y


def maps(depth_est, counts, rng, sigma_e=0.0):
    if sigma_e <= 0:
        B = np.stack([(depth_est >= lo) & (depth_est < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]).astype(float)
        return counts @ B.T
    out = np.zeros((len(counts), len(EDGES) - 1))
    for t in range(len(counts)):
        d = np.repeat(depth_est, counts[t]) + rng.normal(0, sigma_e, counts[t].sum())
        out[t] = np.histogram(d, EDGES)[0]
    return out


def z(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def run(model, seed):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    sess = [make_session(rng, model) for _ in range(S)]
    out = {}
    # within-session counts (position-free reference)
    acc = []
    for depth, counts, y in sess:
        f = np.minimum(np.arange(len(y)) * 5 // len(y), 4); p = np.zeros(len(y), int)
        for k in range(5):
            m = LogisticRegression(C=0.1, max_iter=3000).fit(z(counts[f != k]), y[f != k]); p[f == k] = m.predict(z(counts[f == k]))
        acc.append(np.mean(p == y))
    out["within_counts"] = float(np.mean(acc))

    def cross(feats):
        a = []
        for i in range(S):
            Xtr = np.vstack([z(feats[j]) for j in range(S) if j != i]); ytr = np.concatenate([sess[j][2] for j in range(S) if j != i])
            m = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr, ytr)
            a.append(np.mean(m.predict(z(feats[i])) == sess[i][2]))
        return float(np.mean(a))
    for su in (0.0, 3.0, 5.0, 12.0, 25.0, 50.0, 100.0, 200.0):
        out[f"cross_unit_sigma{int(su)}"] = cross([maps(np.clip(d + rng.normal(0, su, N), 0, 999.9), c, rng) for d, c, _ in sess])
    out["cross_unit_permuted"] = cross([maps(rng.permutation(d), c, rng) for d, c, _ in sess])
    for se in (12.0, 25.0, 50.0, 100.0):
        out[f"cross_event_sigma{int(se)}"] = cross([maps(d, c, rng, sigma_e=se) for d, c, _ in sess])
    out["cross_total_count"] = cross([c.sum(1, keepdims=True).astype(float) for _, c, _ in sess])
    return out


if __name__ == "__main__":
    res = {"protocol": __doc__, "chance": 1.0 / K}
    for model, deltas in (("M1", (20.0, 50.0, 100.0, 250.0)), ("M2", (50.0,))):
        for dl in deltas:
            DELTA = dl; EDGES = np.arange(0, 1000 + 1e-6, dl / 2)
            reps = [run(model, 20271701 + r) for r in range(5)]
            key = f"{model}_delta{int(dl)}"
            res[key] = {k: {"mean": float(np.mean([x[k] for x in reps])), "sd": float(np.std([x[k] for x in reps]))} for k in reps[0]}
            print(key, {k: round(v["mean"], 3) for k, v in res[key].items()}, flush=True)
    Path("reports/phase13/sim13_localization_link.json").write_text(json.dumps(res, indent=1) + "\n")
