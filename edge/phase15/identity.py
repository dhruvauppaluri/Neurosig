"""Phase 15 Track C: online identity with abstention accounting and coverage-oriented decision rules.

OnlineUnits15 extends the frozen Phase 14 OU-P14 (edge/phase14/online_units.py, unchanged) with:
  reasons     every non-assignment is labelled: weak (SNR below snr_assign), no_candidate (no unit near the channel),
              gray (best distance between tau_assign and tau_new), low_gain (template explains too little energy),
              ambiguous (runner-up too close), tentative (assigned to a unit that never became active / expired),
              new_unit (opened a tentative unit)
  lookback    when a tentative unit becomes active, its buffered member events are kept (Phase 14 already did) AND the
              recent unassigned events (last lookback_s seconds, same neighbourhood) are re-scored against the new
              prototype with the same rule; a bounded decision delay of lookback_s
  llr         optional probabilistic low-SNR assignment: for events below snr_assign, assign iff the log-likelihood
              ratio of the best unit against the null (no template) AND against the runner-up both exceed llr_min
              (Gaussian noise-normalized residual model)
  stale_s     tentative lifetime (Phase 14: 6 s) is a parameter; sparse units need longer
Sorter labels are never read here.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from edge.phase14.online_units import SEG, SHIFTS, T0, OnlineUnits, Params, Unit


class Params15(Params):
    pass


class OnlineUnits15(OnlineUnits):
    def __init__(self, E, params: Params | None = None, lookback_s: float = 0.0, llr_min: float = np.inf,
                 low_snr_floor: float = 4.0, tau_rel: float = 0.0, refractory_ms: float = 0.0, peak_um: float = np.inf,
                 margin_rel: float = 0.0, split_every: int = 0, split_dprime: float = 4.0, split_min_frac: float = 0.2):
        super().__init__(E, params)
        self.split_every, self.split_dprime, self.split_min_frac = split_every, split_dprime, split_min_frac
        self.mbuf = {}                     # uid -> deque of (event index, flattened aligned segment)
        self.relabel = {}                  # event index -> uid after a split
        self.tau_rel = tau_rel; self.refr = refractory_ms / 1000.0; self.peak_um = peak_um; self.margin_rel = margin_rel
        self.reason = []
        self.lookback_s, self.llr_min, self.low_snr_floor = lookback_s, llr_min, low_snr_floor
        self.buf = deque()
        self.final_override = {}

    def _score(self, wave, ch):
        out = []
        for u in self.units:
            if u.alive and np.linalg.norm(self.E[u.peak] - self.E[ch]) <= self.p.radius:
                d, k, d0 = self._dist(self._segs(wave, u.contacts), u.mu, with_null=True)
                out.append((d, k, u, d0))
        out.sort(key=lambda z: z[0])
        return out

    def add(self, wave, amp, ch, t, snr, key):
        self.n_seen += 1
        idx = len(self.assign)
        p = self.p
        sc = self._score(wave, ch) if snr >= min(p.snr_assign, self.low_snr_floor if np.isfinite(self.llr_min) else p.snr_assign) else []
        out, why = -1, None
        if snr < p.snr_assign:
            why = "weak"
            if sc and np.isfinite(self.llr_min) and sc[0][2].active:
                d, k, u, d0 = sc[0]; n = u.mu.size
                llr_null = 0.5 * (d0 - d) * n
                run = next((z for z in sc[1:] if z[2].uid != u.uid), None)
                llr_run = 0.5 * (run[0] - d) * n if run else np.inf
                if d <= self._thr(u) and llr_null >= self.llr_min and llr_run >= self.llr_min:
                    self._update_lowsnr(u, key); out, why = u.uid, None
        elif not sc:
            why = "no_candidate"
        elif sc[0][0] <= self._thr(sc[0][2]):
            d, k, u, d0 = sc[0]
            runner = next((z for z in sc[1:] if z[2].uid != u.uid), None)
            relaxed = d > p.tau_assign
            if (d0 - d) * u.mu.size < p.min_gain:
                why = "low_gain"
            elif runner is not None and (runner[0] - d < max(p.margin, self.margin_rel * d) or (runner[0] - d) * u.mu.size < p.margin_abs):
                out, why = -2, "ambiguous"
            elif self.refr > 0 and u.active and 0 <= t - u.last_t < self.refr:
                out, why = -2, "refractory"
            elif relaxed and np.linalg.norm(self.E[u.peak] - self.E[ch]) > self.peak_um:
                why = "peak_mismatch"
            else:
                was_active = u.active
                seg = self._segs(wave, u.contacts)[k]
                self._update(u, seg, amp, t, key)
                out = u.uid
                if self.split_every:
                    b = self.mbuf.setdefault(u.uid, deque(maxlen=200)); b.append((idx, seg.ravel().copy(), tuple(u.contacts)))
                    if u.active and len(b) >= 60 and u.n % self.split_every == 0:
                        self._try_split(u, t)
                if not was_active and u.active and self.lookback_s > 0:
                    self._lookback(u, t)
        elif sc[0][0] >= max(p.tau_new, self._thr(sc[0][2]) + (p.tau_new - p.tau_assign)):
            why = "no_match"
        else:
            why = "gray"
        if out == -1 and why in ("no_candidate", "no_match") and snr >= p.snr_new:
            c = self.nb[ch]
            u = Unit(len(self.units), ch, c, wave[T0 + SEG[0]:T0 + SEG[1]][:, c].astype(float), np.asarray(amp, float).copy(), 1, False, t, t, [key])
            self.units.append(u); out, why = u.uid, None
        self._expire(t)
        if self.n_seen % p.merge_every == 0:
            self._merge()
        self.assign.append(out); self.reason.append(why)
        if out < 0 and self.lookback_s > 0:
            self.buf.append((t, idx, wave, ch, snr))
        while self.buf and t - self.buf[0][0] > self.lookback_s:
            self.buf.popleft()
        return out

    def _try_split(self, u: Unit, t: float):
        """Online split: 2-means on the unit's recent members (same contact set). Split when both clusters hold
        >= split_min_frac of members and their separation along the discriminant exceeds split_dprime (noise units)."""
        b = [(i, x) for i, x, c in self.mbuf[u.uid] if c == tuple(u.contacts)]
        if len(b) < 60:
            return
        Xm = np.stack([x for _, x in b])
        Xc = Xm - Xm.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:3].T
        a = np.argmin(np.abs(Z[:, :1] - np.percentile(Z[:, 0], [25, 75])[None]), axis=1)
        for _ in range(20):
            c0, c1 = Z[a == 0].mean(0) if np.any(a == 0) else Z.mean(0), Z[a == 1].mean(0) if np.any(a == 1) else Z.mean(0)
            a = (np.linalg.norm(Z - c1, axis=1) < np.linalg.norm(Z - c0, axis=1)).astype(int)
        f = a.mean()
        if min(f, 1 - f) < self.split_min_frac:
            return
        m0, m1 = Xm[a == 0].mean(0), Xm[a == 1].mean(0)
        w = (m1 - m0); nw = np.linalg.norm(w)
        if nw <= 0:
            return
        proj = Xm @ (w / nw)
        s = np.sqrt(0.5 * (proj[a == 0].var() + proj[a == 1].var())) + 1e-6
        if abs(proj[a == 1].mean() - proj[a == 0].mean()) / s < self.split_dprime:
            return
        shape = u.mu.shape
        keep, new = (0, 1) if (a == 0).sum() >= (a == 1).sum() else (1, 0)
        u.mu = (m0 if keep == 0 else m1).reshape(shape)
        v = Unit(len(self.units), u.peak, u.contacts, (m1 if keep == 0 else m0).reshape(shape), u.amp.copy(), int((a == new).sum()), True, t, t, [])
        self.units.append(v)
        for (i, _), lab in zip(b, a):
            if lab == new:
                self.relabel[i] = v.uid
        self.mbuf[u.uid] = deque([e for e, lab in zip(self.mbuf[u.uid], a) if lab == keep], maxlen=200)

    def _thr(self, u: Unit) -> float:
        """Assignment distance threshold: noise-only tolerance tau_assign, widened in proportion to the template's
        energy per sample (waveform variability scales with amplitude; tau_rel = 0 reproduces OU-P14)."""
        return max(self.p.tau_assign, 1.0 + self.tau_rel * float(np.mean(u.mu * u.mu)))

    def _update_lowsnr(self, u: Unit, key):
        """Low-SNR events join the unit's member list but do not update its prototype (no template corruption)."""
        u.members.append(key)

    def _lookback(self, u: Unit, t):
        for (tb, idx, wave, ch, snr) in list(self.buf):
            if np.linalg.norm(self.E[u.peak] - self.E[ch]) > self.p.radius or snr < self.p.snr_assign:
                continue
            d, k, d0 = self._dist(self._segs(wave, u.contacts), u.mu, with_null=True)
            if d <= self._thr(u) and (d0 - d) * u.mu.size >= self.p.min_gain:
                self.final_override[idx] = u.uid

    def final_labels(self) -> np.ndarray:
        lab = super().final_labels()
        for idx, uid in self.relabel.items():
            if lab[idx] >= 0:
                lab[idx] = self.resolve(uid)
        for idx, uid in self.final_override.items():
            if lab[idx] < 0:
                r = self.resolve(uid)
                if self.units[r].active and self.units[r].alive:
                    lab[idx] = r
        return lab

    def reasons_final(self) -> dict:
        """Abstention breakdown over all events (final labels): assigned, tentative-never-active, and each reason."""
        lab = self.final_labels()
        out = {"assigned": float(np.mean(lab >= 0))}
        cats = []
        for l, a, r in zip(lab, self.assign, self.reason):
            if l >= 0:
                cats.append("assigned")
            elif a >= 0:
                cats.append("tentative_never_active")
            else:
                cats.append(r or "other")
        v, c = np.unique(cats, return_counts=True)
        out.update({k: float(n / len(cats)) for k, n in zip(v, c)})
        return out, np.array(cats)
