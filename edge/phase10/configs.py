"""Phase 10 configuration P10 = frozen P9 + dead-contact handling by exclusion instead of interpolation (X1/X2).

If no contact is detected as dead (data-only rule ``edge.phase9.improved.dead_channels``), P10 returns exactly the P9
engine's estimate. Otherwise the P9 engine is rebuilt for the live contacts (cached per dead set) and run on the
reduced snippet, so a flat contact is treated as missing rather than reconstructed by spatial interpolation.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from edge.phase8.registry import config_hash
from edge.phase9.configs import P9_HASH, P9_PARAMS, p9_factory
from edge.phase9.harness import sub_array, sub_snippet
from edge.phase9.improved import dead_channels
from edge.types import ElectrodeArray, LocalizationEstimate, Snippet

P10_PARAMS = {**P9_PARAMS, "base_hash": P9_HASH, "dead_contacts": "exclude (rebuild engine on live contacts)",
              "max_cached_masks": 16}
P10_HASH = config_hash(P10_PARAMS)


class DeadAwareLocalizer:
    def __init__(self, array: ElectrodeArray, name: str, full_engine, max_cache: int = 16) -> None:
        self.array, self.name, self.full = array, name, full_engine
        self.method = getattr(full_engine, "method", name)
        self._cache: OrderedDict = OrderedDict()
        self.max_cache = max_cache

    def _engine(self, keep: np.ndarray):
        key = keep.tobytes()
        if key not in self._cache:
            self._cache[key] = p9_factory(sub_array(self.array, keep))[self.name]
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        dead = dead_channels(np.asarray(snippet.voltages_uv, float), snippet.n_pre)
        if not dead.any() or dead.sum() > len(dead) - 4:
            return self.full.estimate(snippet)
        keep = ~dead
        return self._engine(keep).estimate(sub_snippet(snippet, keep))


def p10_factory(array: ElectrodeArray) -> dict:
    return {k: DeadAwareLocalizer(array, k, e, P10_PARAMS["max_cached_masks"]) for k, e in p9_factory(array).items()}
