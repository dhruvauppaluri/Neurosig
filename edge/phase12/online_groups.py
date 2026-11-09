"""Phase 12.2b: conservative, sorter-free online grouping of detector events for unit-level pooling.

Detector-v2.0 events arrive in time order. Each event's feature is its local raw-CAR waveform (contacts within 40 um
of the detector channel, -0.3 .. +0.6 ms, baseline removed), compared only with groups whose anchor channel is the
same or adjacent (<= 40 um). An event joins the most similar group when the normalized correlation with the group's
running-mean template is >= RHO and its amplitude ratio is within [0.5, 2]; otherwise it opens a new tentative group.
Groups with < MIN_GROUP members are 'unassigned' (reported, never dropped from the stream counts). No sorter output is
used to form groups; sorter labels are used only afterwards, to measure purity on matched events.
Thresholds were fixed before evaluation (RHO = 0.9, conservative by design: purity is preferred over completeness).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RHO = 0.90
RADIUS = 40.0
WIN = (-9, 18)
MIN_GROUP = 10
MAX_TEMPLATE_N = 200


@dataclass
class Group:
    gid: int
    anchor: int
    template: np.ndarray        # [T, n_local] running mean
    contacts: np.ndarray
    n: int = 0
    members: list = field(default_factory=list)


def local_feature(rc: np.ndarray, s: int, contacts: np.ndarray) -> np.ndarray | None:
    if s - 40 < 0 or s + WIN[1] > rc.shape[0]:
        return None
    base = np.median(rc[s - 40:s - 12][:, contacts], axis=0)
    return rc[s + WIN[0]:s + WIN[1]][:, contacts] - base[None]


class OnlineGrouper:
    def __init__(self, E: np.ndarray, rho: float = RHO):
        self.E, self.rho = E, rho
        self.nb = [np.flatnonzero(np.linalg.norm(E[:, :2] - E[c, :2], axis=1) <= RADIUS) for c in range(len(E))]
        self.groups: list[Group] = []
        self.near = [np.flatnonzero(np.linalg.norm(E[:, :2] - E[c, :2], axis=1) <= RADIUS) for c in range(len(E))]
        self.by_anchor: dict[int, list[Group]] = {}

    def add(self, rc: np.ndarray, s: int, ch: int, key) -> int | None:
        f = local_feature(rc, s, self.nb[ch])
        if f is None:
            return None
        best, bg = -1.0, None
        cands = [g for c in self.near[ch] for g in self.by_anchor.get(int(c), [])]
        for g in cands:
            ff = local_feature(rc, s, g.contacts)
            if ff is None:
                continue
            a, b = ff.ravel(), g.template.ravel()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            if den <= 0:
                continue
            r = float(a @ b / den); ratio = np.linalg.norm(a) / np.linalg.norm(b)
            if r > best and 0.5 <= ratio <= 2.0:
                best, bg = r, (g, ff)
        if bg is not None and best >= self.rho:
            g, ff = bg
            w = 1.0 / min(g.n + 1, MAX_TEMPLATE_N)
            g.template = (1 - w) * g.template + w * ff
            g.n += 1; g.members.append(key)
            return g.gid
        g = Group(len(self.groups), ch, f.copy(), self.nb[ch], 1, [key])
        self.groups.append(g); self.by_anchor.setdefault(int(ch), []).append(g)
        return g.gid
