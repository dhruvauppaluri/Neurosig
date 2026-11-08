"""Frozen Phase 11 configuration P11 (decided on development / selection data only, before the M-P final windows).

Single-source engines: P10 (frozen P9 engines, lambda = 20 um line source; 11B found no forward model that improved
real-data stability) with the dead-contact mask cache bounded to 4 masks (11G; outputs unchanged).
Dead-contact fallback (11D, chosen on dev by the pre-registered rule): when the data show dead contacts, the P9
(interpolate) and P10 (exclude) estimates are both computed; if they disagree by more than 80 um the estimate is
returned unavailable with reason 'dead-contact ambiguity'; otherwise the P10 estimate is returned.
Quality flag: events whose best nearby contact has no measurable trough (SNR < 4 on the frozen preprocessing) are
reported with reason prefix 'weak_trough' but NOT removed (11A/11C).
Overlap (11E): Phase 9/10 K<=2 space-time pipeline, dead contacts masked, boundary veto only at the axial ends (OV-A);
K = 3 remains experimental and off. Calibration (11F): Mondrian conformal, stratum x SNR group x dropout, fitted on
p11dev (reports/phase11/calibration_p11.json).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from edge.phase8.registry import config_hash
from edge.phase9.configs import p9_factory
from edge.phase9.harness import sub_array, sub_snippet
from edge.phase9.improved import dead_channels
from edge.phase10.configs import P10_HASH
from edge.localize.common import finish_unavailable
from edge.types import LocalizationEstimate, Snippet

P11_PARAMS = {"base": f"P10 {P10_HASH}", "forward_model": "line source lambda=20 (unchanged)", "max_cached_masks": 4,
              "dead_fallback": {"rule": "disagree>k->reject", "k_um": 80.0}, "weak_trough_flag_snr": 4.0,
              "overlap_rule": {"axial_only_boundary": True, "normalized": False, "k3": False},
              "calibration": "mondrian stratum x snr_group(8,16) x dropout, p11dev"}
P11_HASH = config_hash(P11_PARAMS)


class P11Localizer:
    def __init__(self, array, name, max_cache=4, k_um=80.0):
        self.array, self.name, self.k = array, name, k_um
        self.p9 = p9_factory(array)[name]
        self.method = getattr(self.p9, "method", name)
        self._cache: OrderedDict = OrderedDict(); self.max_cache = max_cache

    def _masked(self, keep):
        key = keep.tobytes()
        if key not in self._cache:
            self._cache[key] = p9_factory(sub_array(self.array, keep))[self.name]
            if len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def estimate(self, sn: Snippet) -> LocalizationEstimate:
        dead = dead_channels(np.asarray(sn.voltages_uv, float), sn.n_pre)
        if not dead.any() or dead.sum() > len(dead) - 4:
            return self.p9.estimate(sn)
        keep = ~dead
        m = self._masked(keep).estimate(sub_snippet(sn, keep))
        i = self.p9.estimate(sn)
        if m.available and i.available and np.linalg.norm(np.array(m.xyz_um[:2]) - np.array(i.xyz_um[:2])) > self.k:
            import time
            return finish_unavailable(method=self.method, snippet=sn, t_start=time.perf_counter(),
                                      reason="dead-contact ambiguity: interpolate/exclude estimates disagree")
        return m


def p11_factory(array):
    return {k: P11Localizer(array, k, P11_PARAMS["max_cached_masks"], P11_PARAMS["dead_fallback"]["k_um"])
            for k in p9_factory(array)}
