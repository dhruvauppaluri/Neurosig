"""Phase 11D: dead-contact tail repair. Records, per event and unknown-dropout mask, the full-array estimate, the
P9 (interpolate) and P10 (exclude) estimates, and data-only features available at run time:
  n_dead_near   dead contacts within 60 um of the P10 estimate
  n_live_near   live contacts within 60 um of the P10 estimate
  gap_um        distance from the P10 estimate to the nearest live contact
  extrap        P10 estimate outside the live-contact axial span (+-10 um)
  disagree_um   |P10 - P9| (the two dead-contact strategies disagree)
Policies are chosen on development windows only and evaluated on selection windows.
"""

from __future__ import annotations

import numpy as np

from edge.phase9.harness import make_mask
from edge.phase9.improved import dead_channels
from edge.types import Snippet

CONDITIONS = (("random", 0.75), ("random", 0.5), ("clustered", 0.75))


def _zero(sn, keep):
    v = np.asarray(sn.voltages_uv, float).copy(); v[:, ~keep] = 0.0
    return Snippet(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um,
                   sn.peak_channel, sn.ground_truth_xyz_um)


def collect(events, p9_factory, p10_factory, engines=("beamform", "weighted_scan", "robust_scan"), seed=20271111,
            max_per_window=25):
    rng = np.random.default_rng(seed)
    rows = []
    for ew in events:
        array = ew["array"]; E = array.xyz_um()
        f9, f10 = p9_factory(array), p10_factory(array)
        for i, sn in enumerate(ew["snippets"][:max_per_window]):
            full = {}
            for n in engines:
                r = f10[n].estimate(sn)
                full[n] = np.array(r.xyz_um[:2]) if r.available else None
            for kind, frac in CONDITIONS:
                keep = make_mask(array.n_channels, frac, kind, rng)
                z = _zero(sn, keep)
                live = ~dead_channels(np.asarray(z.voltages_uv, float), z.n_pre)
                for n in engines:
                    if full[n] is None:
                        continue
                    a, b = f9[n].estimate(z), f10[n].estimate(z)
                    if not (a.available and b.available):
                        rows.append({"engine": n, "cond": f"{kind}_{frac}", "session": ew["window"]["session"], "failed": True}); continue
                    xi, xm = np.array(a.xyz_um[:2]), np.array(b.xyz_um[:2])
                    d = np.linalg.norm(E[:, :2] - xm, axis=1)
                    ys = E[live, 1]
                    rows.append({"engine": n, "cond": f"{kind}_{frac}", "session": ew["window"]["session"], "failed": False,
                                 "disp_p9": float(np.linalg.norm(xi - full[n])), "disp_p10": float(np.linalg.norm(xm - full[n])),
                                 "n_dead_near": int(np.sum((d <= 60) & ~live)), "n_live_near": int(np.sum((d <= 60) & live)),
                                 "gap_um": float(d[live].min()) if live.any() else 1e9,
                                 "extrap": bool(xm[1] < ys.min() - 10 or xm[1] > ys.max() + 10) if live.any() else True,
                                 "disagree_um": float(np.linalg.norm(xm - xi)), "n_dead": int((~live).sum())})
    return rows


POLICIES = {
    "P10": lambda r, k: "mask",
    "P9": lambda r, k: "interp",
    "dead_near>=k->interp": lambda r, k: "interp" if r["n_dead_near"] >= k else "mask",
    "live_near<k->interp": lambda r, k: "interp" if r["n_live_near"] < k else "mask",
    "extrap->interp": lambda r, k: "interp" if r["extrap"] else "mask",
    "disagree>k->reject": lambda r, k: "reject" if r["disagree_um"] > k else "mask",
    "disagree>k->interp": lambda r, k: "interp" if r["disagree_um"] > k else "mask",
}


def apply(rows, policy, k):
    out, rej = [], 0
    for r in rows:
        if r["failed"]:
            continue
        c = POLICIES[policy](r, k)
        if c == "reject":
            rej += 1; continue
        out.append(r["disp_p9"] if c == "interp" else r["disp_p10"])
    n = len([r for r in rows if not r["failed"]])
    v = np.array(out)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "accepted": 1 - rej / max(n, 1), "n": n}
