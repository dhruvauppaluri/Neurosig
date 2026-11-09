"""Phase 14.4-14.5: sorter-free online unit association, drift-aware prototypes and unit localization.

Stages (one detector event at a time, in time order; no sorter output is ever read):
  A  detect      frozen detector-v2.0 (cached by edge/phase14/detcache.py)
  B  extract     noise-normalized filtered waveform [-15, +30) samples, E3 trough amplitudes
  C  propose     candidate units = prototypes whose peak contact lies within RADIUS of the event's detector channel;
                 distance d2 = min over shift tau in [-2, 2] and gain a in [A_LO, A_HI] of ||x_tau - a mu||^2 / n on the
                 unit's own contact set (contacts within RADIUS of its peak contact) over samples [-10, +20)
                 (noise-normalized, so d2 ~ 1 for a perfect template plus white noise)
  D  decide      assign if best d2 <= TAU_ASSIGN and the runner-up unit is worse by >= MARGIN (else 'ambiguous',
                 left unassigned); if best d2 >= TAU_NEW and SNR >= SNR_NEW open a TENTATIVE unit; everything
                 else stays unassigned (gray zone). A tentative unit becomes ACTIVE after N_INIT assignments;
                 tentative units that stay below N_INIT for STALE_S seconds are discarded (their events unassigned).
  E  update      prototype mu <- (1 - alpha) mu + alpha x_tau*, alpha = max(1 / n, ALPHA_MIN): exponential forgetting
                 so slow drift is followed; the unit's amplitude vector (E3) and position use the same weights
  F  drift       unit position = FastLine20 fit of the running amplitude prototype, re-evaluated on update; a
                 position change is reported with the pooled-count-based uncertainty, separately from identity
  merge          every MERGE_EVERY events, active units with overlapping contact sets whose prototypes are within
                 TAU_MERGE of each other (both directions, after alignment) are merged (duplicate suppression)
Parameters are set on development sessions only (frozen in reports/phase14/frozen_p14.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T0 = 15                      # trough index inside the cached [-15, +30) waveform
SEG = (-10, 20)
SHIFTS = (-2, -1, 0, 1, 2)


@dataclass
class Params:
    radius: float = 60.0
    tau_assign: float = 1.8
    tau_new: float = 3.0
    margin: float = 0.25
    snr_new: float = 6.0
    n_init: int = 8
    stale_s: float = 6.0
    alpha_min: float = 0.02
    a_lo: float = 0.6
    a_hi: float = 1.6
    tau_merge: float = 1.6
    merge_every: int = 400
    min_gain: float = 0.0           # required energy explained by the template, (d_null - d) * n (noise units^2)
    margin_abs: float = 0.0         # required runner-up gap in energy units, (d_2 - d_1) * n
    snr_assign: float = 0.0         # events below this detector SNR are never assigned (reported as unassigned)
    relink_quiet_s: float = 0.0     # > 0: a newly activated unit inherits the identity of a SILENT active unit
                                    # (no event for this long) on overlapping contacts whose prototype matches
    relink_tau: float = 3.0


@dataclass
class Unit:
    uid: int
    peak: int
    contacts: np.ndarray
    mu: np.ndarray                    # [30, |contacts|]
    amp: np.ndarray                   # [C] running E3 amplitude prototype
    n: int = 1
    active: bool = False
    born_t: float = 0.0
    last_t: float = 0.0
    members: list = field(default_factory=list)
    alive: bool = True
    merged_into: int = -1


class OnlineUnits:
    def __init__(self, E: np.ndarray, params: Params | None = None):
        self.E = np.asarray(E, float)[:, :2]
        self.p = params or Params()
        self.nb = [np.flatnonzero(np.linalg.norm(self.E - self.E[c], axis=1) <= self.p.radius) for c in range(len(self.E))]
        self.units: list[Unit] = []
        self.assign: list[int] = []           # per event: unit id or -1 (unassigned) / -2 (ambiguous)
        self.n_seen = 0

    # ---------------------------------------------------------------- distances
    def _segs(self, wave: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        """[len(SHIFTS), 30, |contacts|] shifted event segments."""
        return np.stack([wave[T0 + SEG[0] + s:T0 + SEG[1] + s][:, contacts] for s in SHIFTS]).astype(float)

    def _dist(self, X: np.ndarray, mu: np.ndarray, with_null: bool = False):
        mm = float(np.sum(mu * mu))
        if mm <= 0:
            return (np.inf, 0, 0.0) if with_null else (np.inf, 0)
        a = np.clip(np.einsum("ktc,tc->k", X, mu) / mm, self.p.a_lo, self.p.a_hi)
        r = X - a[:, None, None] * mu[None]
        d = np.sum(r * r, axis=(1, 2)) / mu.size
        k = int(np.argmin(d))
        if with_null:
            return float(d[k]), k, float(np.sum(X[k] * X[k]) / mu.size)
        return float(d[k]), k

    # ---------------------------------------------------------------- stream
    def add(self, wave: np.ndarray, amp: np.ndarray, ch: int, t: float, snr: float, key) -> int:
        self.n_seen += 1
        cands = [u for u in self.units if u.alive and np.linalg.norm(self.E[u.peak] - self.E[ch]) <= self.p.radius]
        scored = []
        for u in cands:
            d, k, d0 = self._dist(self._segs(wave, u.contacts), u.mu, with_null=True)
            scored.append((d, k, u, d0))
        scored.sort(key=lambda z: z[0])
        out = -1
        if snr < self.p.snr_assign:
            scored = []
        if scored and scored[0][0] <= self.p.tau_assign and (scored[0][3] - scored[0][0]) * scored[0][2].mu.size >= self.p.min_gain:
            d, k, u, _ = scored[0]
            runner = next((z for z in scored[1:] if z[2].uid != u.uid), None)
            if runner is not None and (runner[0] - d < self.p.margin or (runner[0] - d) * u.mu.size < self.p.margin_abs):
                out = -2
            else:
                self._update(u, self._segs(wave, u.contacts)[k], amp, t, key)
                out = u.uid
        elif (not scored or scored[0][0] >= self.p.tau_new) and snr >= self.p.snr_new:
            c = self.nb[ch]
            u = Unit(len(self.units), ch, c, wave[T0 + SEG[0]:T0 + SEG[1]][:, c].astype(float), np.asarray(amp, float).copy(),
                     1, False, t, t, [key])
            self.units.append(u); out = u.uid
        self._expire(t)
        if self.n_seen % self.p.merge_every == 0:
            self._merge()
        self.assign.append(out)
        return out

    def _update(self, u: Unit, x: np.ndarray, amp: np.ndarray, t: float, key):
        u.n += 1
        a = max(1.0 / u.n, self.p.alpha_min)
        u.mu = (1 - a) * u.mu + a * x
        u.amp = (1 - a) * u.amp + a * np.asarray(amp, float)
        u.last_t = t; u.members.append(key)
        if not u.active and u.n >= self.p.n_init:
            u.active = True
            if self.p.relink_quiet_s > 0:
                self._relink(u, t)
        # keep the contact set centred on the prototype's current peak contact (slow drift across contacts)
        pk = int(u.contacts[int(np.argmin(u.mu.min(axis=0)))])
        if pk != u.peak and u.active and u.n % 20 == 0:
            new = self.nb[pk]
            common = np.isin(new, u.contacts)
            mu = np.zeros((u.mu.shape[0], len(new)))
            idx = {int(c): i for i, c in enumerate(u.contacts)}
            for j, c in enumerate(new):
                if int(c) in idx:
                    mu[:, j] = u.mu[:, idx[int(c)]]
            if common.mean() >= 0.5:
                u.peak, u.contacts, u.mu = pk, new, mu

    def _relink(self, u: Unit, t: float):
        """Identity continuity across abrupt waveform change (drift): merge the new unit INTO a silent old unit."""
        best = None
        for o in self.units:
            if o is u or not (o.alive and o.active) or t - o.last_t < self.p.relink_quiet_s:
                continue
            if np.linalg.norm(self.E[o.peak] - self.E[u.peak]) > self.p.radius:
                continue
            c = np.intersect1d(o.contacts, u.contacts)
            if len(c) < 3:
                continue
            mo, mu = o.mu[:, np.searchsorted(o.contacts, c)], u.mu[:, np.searchsorted(u.contacts, c)]
            X = np.stack([np.roll(mu, s, axis=0) for s in SHIFTS])
            d, _ = self._dist(X, mo)
            if d <= self.p.relink_tau and (best is None or d < best[0]):
                best = (d, o)
        if best is not None:
            o = best[1]
            o.mu, o.contacts, o.peak, o.amp = u.mu, u.contacts, u.peak, u.amp      # the new shape replaces the old one
            o.n += u.n; o.members += u.members; o.last_t = t
            u.alive = False; u.merged_into = o.uid

    def _expire(self, t: float):
        for u in self.units:
            if u.alive and not u.active and t - u.born_t > self.p.stale_s:
                u.alive = False

    def _merge(self):
        act = [u for u in self.units if u.alive and u.active]
        for i, a in enumerate(act):
            if not a.alive:
                continue
            for b in act[i + 1:]:
                if not b.alive or np.linalg.norm(self.E[a.peak] - self.E[b.peak]) > self.p.radius:
                    continue
                c = np.intersect1d(a.contacts, b.contacts)
                if len(c) < 3:
                    continue
                ia = np.searchsorted(a.contacts, c); ib = np.searchsorted(b.contacts, c)
                ma, mb = a.mu[:, ia], b.mu[:, ib]
                X = np.stack([np.roll(ma, s, axis=0) for s in SHIFTS])
                d1, _ = self._dist(X, mb)
                X = np.stack([np.roll(mb, s, axis=0) for s in SHIFTS])
                d2, _ = self._dist(X, ma)
                if max(d1, d2) <= self.p.tau_merge * max(1.0, 0.05 * float(np.mean(ma ** 2 + mb ** 2))):
                    keep, drop = (a, b) if a.n >= b.n else (b, a)
                    w = drop.n / (keep.n + drop.n)
                    keep.amp = (1 - w) * keep.amp + w * drop.amp
                    keep.n += drop.n; keep.members += drop.members
                    drop.alive = False; drop.merged_into = keep.uid

    def resolve(self, uid: int) -> int:
        while uid >= 0 and self.units[uid].merged_into >= 0:
            uid = self.units[uid].merged_into
        return uid

    def final_labels(self) -> np.ndarray:
        """Per-event final label: active (possibly merged) unit id, else -1 (unassigned / ambiguous / tentative)."""
        out = []
        for u in self.assign:
            if u < 0:
                out.append(-1); continue
            r = self.resolve(u)
            out.append(r if (self.units[r].active and self.units[r].alive) else -1)
        return np.array(out)

    def online_labels(self) -> np.ndarray:
        """Label at decision time (tentative units count as unassigned): what a streaming consumer saw."""
        return np.array([u if (u >= 0) else -1 for u in self.assign])


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(labels: np.ndarray, sorter: np.ndarray, snr: np.ndarray, quality: np.ndarray | None = None) -> dict:
    """Explicit assignment metrics against sorter labels (evaluation only; sorter = -1 for unmatched events).

    precision   assigned & matched events whose sorter unit equals the majority sorter unit of their online unit
    coverage    assigned / all detector events; also on the predefined high-quality subset (matched to a sorter unit
                with label == 1 and SNR >= 8)
    merge rate  online units (>= 5 matched events) whose second-largest sorter unit holds >= 20% of their matched events
    fragmentation  online units per sorter unit (sorter units with >= 5 assigned matched events)
    duplicates  sorter units with >= 5 events in each of >= 2 online units
    ARI         adjusted Rand index on assigned matched events (secondary)
    """
    from collections import Counter
    labels = np.asarray(labels); sorter = np.asarray(sorter)
    asg = labels >= 0; m = sorter >= 0
    both = asg & m
    maj = {}
    for u in np.unique(labels[both]):
        maj[int(u)] = Counter(sorter[both & (labels == u)].tolist()).most_common(1)[0][0]
    correct = np.array([maj[int(l)] == s for l, s in zip(labels[both], sorter[both])])
    merged = frag = 0; units_ok = 0
    for u in np.unique(labels[both]):
        c = Counter(sorter[both & (labels == u)].tolist()).most_common()
        n = sum(v for _, v in c)
        if n >= 5:
            units_ok += 1
            if len(c) > 1 and c[1][1] / n >= 0.2:
                merged += 1
    per_sorter = {}
    for s in np.unique(sorter[both]):
        c = Counter(labels[both & (sorter == s)].tolist())
        big = [k for k, v in c.items() if v >= 5]
        if big:
            per_sorter[int(s)] = len(big)
    hq = m & (snr >= 8) if quality is None else (m & (snr >= 8) & (quality >= 1))
    out = {"n_events": int(len(labels)), "precision": float(correct.mean()) if len(correct) else np.nan,
           "n_precision_events": int(len(correct)), "coverage_all": float(asg.mean()),
           "coverage_matched": float(asg[m].mean()) if m.any() else np.nan,
           "coverage_hq": float(asg[hq].mean()) if hq.any() else np.nan, "n_hq": int(hq.sum()),
           "precision_hq": float(np.mean([maj.get(int(l)) == s for l, s in zip(labels[hq & asg], sorter[hq & asg])])) if (hq & asg).any() else np.nan,
           "n_units": int(len(np.unique(labels[asg]))), "merge_rate": merged / max(units_ok, 1),
           "fragmentation_mean": float(np.mean(list(per_sorter.values()))) if per_sorter else np.nan,
           "duplicate_units_frac": float(np.mean([v >= 2 for v in per_sorter.values()])) if per_sorter else np.nan,
           "ari": ari(labels[both], sorter[both]) if both.sum() > 2 else np.nan}
    bins = [(0, 6), (6, 8), (8, 12), (12, 1e9)]
    out["by_snr"] = {}
    for lo, hi in bins:
        k = m & (snr >= lo) & (snr < hi)
        kk = k & asg
        out["by_snr"][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = {
            "n": int(k.sum()), "coverage": float(asg[k].mean()) if k.any() else np.nan,
            "precision": float(np.mean([maj.get(int(l)) == s for l, s in zip(labels[kk], sorter[kk])])) if kk.any() else np.nan}
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    from math import comb
    from collections import Counter
    n = len(a)
    if n < 2:
        return np.nan
    cont = Counter(zip(a.tolist(), b.tolist()))
    sa = sum(comb(v, 2) for v in Counter(a.tolist()).values()); sb = sum(comb(v, 2) for v in Counter(b.tolist()).values())
    sab = sum(comb(v, 2) for v in cont.values()); tot = comb(n, 2)
    exp = sa * sb / tot; mx = 0.5 * (sa + sb)
    return float((sab - exp) / (mx - exp)) if mx != exp else 1.0
