"""Phase 12.4: expanded overlap / false-source benchmark and temporal count features.

Two independently specified event families per pool (seeds fixed in SEEDS, pools never mixed):

  syn  the Phase 9/10 line-source generator (edge/phase10/overlap_stress.make_stress) with new seeds: singles, pairs
       over dt and separation bins, triples. Its forward model matches the fitter (optimistic).
  bio  MEArec/NEURON templates (6 L5 cell models; model-mismatched) + real IBL noise (edge/phase12/biosim.py), with
       single-source conditions clean / amplitude jitter (+-30%) / artifact (2-sample impulse on 3 contacts) /
       3 dead contacts, and pairs / triples built by moving a second (third) template axially on the first one's
       contact set. SNR is peak trough over the filtered noise MAD of the source's own peak contact.

Resolvable (fixed before evaluation, the Phase 9 definition): min pairwise in-plane separation >= 60 um, every source
SNR >= 6, amplitude ratio >= 1/3. All pairs and all triples are always reported next to the resolvable subsets.

Temporal / spatiotemporal count features (computed from the conditioned snippet, no truth):
  rank2      s2/s1 of the noise-whitened, common-mode-removed window [-0.3, +0.6] ms (temporal rank)
  n_trough   max over contacts of distinct troughs (>= 5 sigma, >= 4 samples apart) in [-1.0, +1.3] ms
  t_spread   SNR-weighted std (samples) of per-contact trough times over contacts with SNR >= 5
  delta_n    free two-source SSE drop / max(chi2_red(K=1), 1)   (OV-B statistic)
  dshared_n  shared-waveform SSE drop / max(chi2_red(K=1), 1)
  chi1       chi2_red of the K=1 fit
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge.phase10 import overlap_stress as O
from edge.phase12 import biosim as B
from edge.snippets import extract_snippet
from edge.types import ElectrodeArray, ElectrodeSite, Snippet

SEEDS = {"p12dev": 20271401, "p12select": 20271402, "p12final": 20271403}
NOISE_POOL = {"p12dev": "dev", "p12select": "select", "p12final": "final"}
O.SEEDS.update(SEEDS)


@dataclass
class BenchEvent:
    family: str           # syn | bio
    kind: str             # single | pair | triple
    cond: str             # clean | ampjit | artifact | dead | dtX | sepX
    sources: np.ndarray   # [K, 2] truth (lateral, axial)
    snr: tuple
    dt: int
    sep: float
    resolvable: bool
    snippet: Snippet
    tids: tuple = ()          # template ids (bio family), for leave-template-out dictionary evaluation


def _array(E):
    return ElectrodeArray(tuple(ElectrodeSite(i, tuple(map(float, E[i]))) for i in range(len(E))), "bench")


def syn_events(pool: str, n_per_cell: int) -> list[BenchEvent]:
    array, ev = O.make_stress(pool, n_per_cell=n_per_cell)
    out = []
    for fam, L in ev.items():
        for e in L:
            src = np.array([s[:2] for s in e.sources])
            out.append(BenchEvent("syn", e.kind, f"{fam}", src, (e.noise_uv,), e.dt, e.sep_um, e.resolvable, e.snippet))
    return out


def _snr_of(W, t0, sig):
    pk = int(np.argmin(W[t0])); return -W[t0, pk] / sig[pk]


def bio_events(pool: str, n_single_per_cond: int, n_pair_per_bin: int, n_triple: int) -> list[BenchEvent]:
    which = B.tset("final" if pool == "p12final" else "dev")
    tpool = B.template_pool({"p12dev": "dev", "p12select": "select", "p12final": "final"}[pool])
    nb = B.noise_bank(NOISE_POOL[pool]); sig = B.noise_sigma(NOISE_POOL[pool])
    rng = np.random.default_rng(SEEDS[pool] + 7)
    out = []

    def snip(seg, E, ref):
        return Snippet(len(out), 40, ref - 40, 30000.0, tuple(range(len(E))), seg[ref - 40:ref + 60], E, None, None)

    for cond in ("clean", "ampjit", "artifact", "dead"):
        for _ in range(n_single_per_cond):
            k = int(rng.choice(tpool)); snr = float(rng.uniform(5, 20))
            ev = B.make_single(k, snr, nb, sig, rng, amp_jitter=0.3 if cond == "ampjit" else 0.0,
                               artifact=cond == "artifact", dead=3 if cond == "dead" else 0, which=which)
            out.append(BenchEvent("bio", "single", cond, ev.truth, (snr,), 0, 0.0, False, snip(ev.seg, ev.E, ev.ref_sample), (k,)))

    def multi(K, dt_lo, dt_hi, sep_lo, sep_hi, cond):
        k1 = int(rng.choice(tpool))
        W1, E, tr1, t01, pk1 = B.template_view(k1, which)
        snr1 = float(rng.uniform(6, 20))
        seg = B.place(W1 * (snr1 * sig[pk1] / -W1[t01, pk1]), t01, B.CENTER)
        truths, snrs, ts, ks = [tr1], [snr1], [B.CENTER], [k1]
        tm, loc, _, _ = B.templates(which)
        for j in range(1, K):
            k2 = int(rng.choice([t for t in tpool if t != k1]))
            side = float(rng.choice([-1, 1])) if j == 1 else -np.sign(truths[1][1] - tr1[1]) or 1.0
            target = tr1[1] + side * rng.uniform(sep_lo, sep_hi)
            dz = round((target - loc[k2, 2]) / 40.0) * 40.0
            W2, tr2, t02 = B.shifted_on_view(k2, E, dz, which)
            if not np.any(W2):
                return None
            pk2 = int(np.argmin(W2[t02]))
            if -W2[t02, pk2] <= 1e-6:
                return None
            snr2 = snr1 * float(rng.uniform(0.5, 1.0))
            dt = int(rng.integers(dt_lo, dt_hi + 1)) * int(rng.choice([-1, 1]))
            seg = seg + B.place(W2 * (snr2 * sig[pk2] / -W2[t02, pk2]), t02, B.CENTER + dt)
            truths.append(tr2); snrs.append(snr2); ts.append(B.CENTER + dt); ks.append(k2)
        seg = seg + nb[rng.integers(len(nb))].astype(float)
        T = np.array(truths)
        sep = min(float(np.linalg.norm(T[a] - T[b])) for a in range(K) for b in range(a + 1, K))
        # observed SNR on the view (a moved template can lose its peak off the contact set)
        res = bool(sep >= 60 and min(snrs) >= 6 and min(snrs) / max(snrs) >= 1 / 3)
        ref = int(ts[int(np.argmax(snrs))]) + int(rng.integers(-2, 3))
        return BenchEvent("bio", "pair" if K == 2 else "triple", cond, T, tuple(snrs), abs(ts[1] - ts[0]), sep, res, snip(seg, E, ref), tuple(ks))

    for lo, hi in ((0, 0), (1, 3), (4, 9), (10, 20)):
        n = 0
        while n < n_pair_per_bin:
            e = multi(2, lo, hi, 40, 140, f"dt{lo}-{hi}")
            if e is not None:
                out.append(e); n += 1
    n = 0
    while n < n_triple:
        e = multi(3, 1, 9, 60, 120, "triple")
        if e is not None:
            out.append(e); n += 1
    return out


def bench(pool: str, scale: int = 1) -> list[BenchEvent]:
    return syn_events(pool, 20 * scale) + bio_events(pool, 250 * scale, 60 * scale, 60 * scale)


# ------------------------------------------------------------------------------------------------ features
def temporal_features(sn: Snippet, sigma: np.ndarray, mask: np.ndarray | None = None) -> dict:
    v = np.asarray(sn.voltages_uv, float)
    k = sn.n_pre
    live = np.ones(v.shape[1], bool) if mask is None else np.asarray(mask, bool)
    s = np.where(live, sigma, np.inf)
    Wn = (v / s[None])[:, live]
    Wn = Wn - Wn.mean(axis=1, keepdims=True)                     # common mode removed
    T = v.shape[0]
    seg = Wn[max(k - 9, 0):min(k + 18, T)]
    sv = np.linalg.svd(seg, compute_uv=False)
    rank2 = float(sv[1] / max(sv[0], 1e-12))
    a0, a1 = max(k - 30, 0), min(k + 40, T)
    z = v[a0:a1][:, live] / sigma[live][None]
    c0, c1 = max(k - 10 - a0, 0), min(k + 10 - a0, a1 - a0)          # trough-time search: +-10 samples
    n_tr = 0
    tmin, wts = [], []
    for c in range(z.shape[1]):
        x = z[:, c]
        loc = np.flatnonzero((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]) & (x[1:-1] < -5.0)) + 1
        kept = []
        for i in loc[np.argsort(x[loc])]:
            if all(abs(i - j) >= 4 for j in kept):
                kept.append(i)
        n_tr = max(n_tr, len(kept))
        mn = int(np.argmin(x[c0:c1])) + c0
        if -x[mn] >= 5:
            tmin.append(mn); wts.append(-x[mn])
    if len(tmin) >= 2:
        w = np.array(wts); t = np.array(tmin, float); m = np.average(t, weights=w)
        tsp = float(np.sqrt(np.average((t - m) ** 2, weights=w)))
    else:
        tsp = 0.0
    return {"rank2": rank2, "n_trough": n_tr, "t_spread": tsp}


FEATURES = ("log_delta_n", "log_dshared_n", "rank2", "n_trough", "t_spread", "log_chi1",
            "pol2", "hcorr", "log_en_ratio", "sep2_60")


def feature_vector(info: dict, tf: dict) -> np.ndarray:
    return np.array([np.log1p(max(info["delta_norm"], 0.0)), np.log1p(max(info["delta_shared_norm"], 0.0)),
                     tf["rank2"], min(tf["n_trough"], 4), min(tf["t_spread"], 10.0), np.log(max(info["chi1"], 1e-6)),
                     info.get("pol2", 0.0), info.get("hcorr", 0.0), np.log(max(info.get("en_ratio", 1e-6), 1e-6)),
                     min(info.get("sep2", 0.0), 200.0) / 60.0])
