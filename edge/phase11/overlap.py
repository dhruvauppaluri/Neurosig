"""Phase 11E source-count selection variants on top of the frozen Phase 9/10 K<=2 space-time pipeline.

OV-A  boundary veto only at the axial ends of the search box (a source fitted at the lateral limit is kept and
      flagged 'lateral_limit_source' instead of collapsing the pair to one source).
OV-B  misfit-normalized count statistics: delta / max(chi2_red(K=1), 1) and delta_shared / max(chi2_red(K=1), 1),
      thresholds recalibrated on the p11dev pool so the single-source false two-source rate is <= 3%.
K=3   the Phase 10 greedy extension, accepted only if its normalized delta3 exceeds a threshold calibrated on
      p11dev resolvable pairs (false three-source rate <= 3%).
Pools p11dev / p11select / p11final use new seeds; the resolvable definition is unchanged from Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from edge.phase8.pipeline import LocalizationResult, Phase8Localizer, SourceEstimate, _ellipse_area
from edge.phase10 import overlap_stress as O

O.SEEDS.update({"p11dev": 20271201, "p11select": 20271202, "p11final": 20271203})


@dataclass(frozen=True)
class CountRule:
    axial_only_boundary: bool = False
    normalized: bool = False
    tau_two: float | None = None        # used when normalized (else the Phase 9 calibration value)
    tau_shared: float | None = None
    tau_three: float | None = None      # normalized delta3 threshold; None disables K=3


def chi1(fit, Wt):
    return fit.sse1 / max(Wt.size - 2 - Wt.shape[1], 1)


def localize(loc: Phase8Localizer, sn, rule: CountRule, mask=None) -> tuple[LocalizationResult, dict]:
    """Same decision logic as Phase8Localizer.localize, with the rule's count statistic and boundary handling."""
    fit, Wt, sigma = loc.analyze(sn, mask)
    if fit is None:
        return LocalizationResult(sn.event_id, [], 0, 0.0, False, "zero signal", ["no_signal"]), {}
    cal = loc.cal
    c1 = max(chi1(fit, Wt), 1.0)
    d, ds = fit.delta, fit.delta_shared
    t2, ts = cal.tau_two, cal.tau_shared
    if rule.normalized:
        d, ds = d / c1, ds / c1
        t2, ts = rule.tau_two, rule.tau_shared
    flags = []
    two_free = d > t2
    two_shared = (not two_free) and ds > ts
    two = two_free or two_shared
    srcs = fit.sources2 if two_free else (fit.sources_shared if two_shared else fit.sources1)
    if two and np.hypot(srcs[0].x - srcs[1].x, srcs[0].y - srcs[1].y) < cal.min_sep_um:
        two, two_free, two_shared, srcs = False, False, False, fit.sources1; flags.append("merged_close_pair")
    f = loc.fitter
    if two:
        at_y = any(s.y >= f.hi[1] - 1.0 or s.y <= f.lo[1] + 1.0 for s in srcs)
        at_x = any(s.x >= f.hi[0] - 1.0 for s in srcs)
        if at_y or (at_x and not rule.axial_only_boundary):
            two, two_free, two_shared, srcs = False, False, False, fit.sources1; flags.append("boundary_pair_rejected")
        elif at_x:
            flags.append("lateral_limit_source")
    pos = np.array([[s.x, s.y] for s in srcs])
    n3 = False
    d3 = None
    if two and two_free and rule.tau_three is not None:
        p3, d3raw = O.k3_fit(loc, sn, pos)
        d3 = d3raw / c1
        if d3 > rule.tau_three and min(np.linalg.norm(p3[2] - p3[i]) for i in range(2)) >= cal.min_sep_um \
                and not (p3[2, 1] >= f.hi[1] - 1 or p3[2, 1] <= f.lo[1] + 1):
            pos, n3 = p3, True
    cov_all = f.covariance(pos, Wt, sigma)
    n_obs = Wt.size
    K = pos.shape[0]
    chi2_red = (fit.sse2 if two else fit.sse1) / max(n_obs - 2 * K - K * Wt.shape[1], 1)
    if (not two) and d > (cal.tau_gray if not rule.normalized else 0.8 * t2):
        flags.append("possible_unresolved_overlap")
    if two_shared:
        flags.append("shared_waveform_pair")
    if n3:
        flags.append("three_source")
    q = cal.q_shared if two_shared else (cal.q_two if two else (cal.q_flagged if flags else cal.q_clean))
    energies = [s.energy for s in srcs] + ([0.0] if n3 else [])
    order = np.argsort([-e for e in energies])
    out = []
    for k in order:
        c = cov_all[2 * k:2 * k + 2, 2 * k:2 * k + 2]; c = 0.5 * (c + c.T) + 1e-9 * np.eye(2)
        out.append(SourceEstimate((float(pos[k, 0]), float(pos[k, 1]), 0.0), c.tolist(), _ellipse_area(c, q), float(energies[k])))
    unit_area = _ellipse_area(np.asarray(out[0].cov_xy), 1.0)
    reason = ""
    if chi2_red > cal.max_chi2_red:
        reason = f"fit residual chi2_red={chi2_red:.2f} exceeds {cal.max_chi2_red}"
    elif unit_area > cal.max_unit_area_um2:
        reason = f"uncertainty area {unit_area:.0f} um^2 exceeds {cal.max_unit_area_um2:.0f}"
    conf = float(np.clip(1.0 - unit_area / max(min(cal.max_unit_area_um2, 1e6), 1.0), 0.0, 1.0))
    res = LocalizationResult(sn.event_id, out, len(out), conf, reason == "", reason, flags, float(fit.delta), float(chi2_red))
    return res, {"delta_norm": d, "delta_shared_norm": ds, "chi1": c1, "delta3_norm": d3}


def evaluate(loc, ev, rule: CountRule, mask=None, zero=None, dead_as_mask=True):
    """Phase 10 evaluate_event semantics (one-to-one matching, 60 um gate) with this rule."""
    from edge.phase9.improved import condition
    sn = ev.snippet
    if zero is not None:
        v = np.asarray(sn.voltages_uv, float).copy(); v[:, zero] = 0.0
        sn = type(sn)(sn.event_id, sn.t_sample, sn.t0_sample, sn.sample_rate_hz, sn.channel_ids, v, sn.electrode_xyz_um, sn.peak_channel, None)
    if dead_as_mask:
        det = O.detected_dead_mask(sn)
        if det is not None:
            mask = det if mask is None else (mask & det)
        c = condition(sn, baseline=True)
    else:
        c = condition(sn, **O.OV_COND)
    import time as _t
    t0 = _t.perf_counter()
    r, info = localize(loc, c, rule, mask)
    ms = (_t.perf_counter() - t0) * 1e3
    est = np.array([s.xyz_um[:2] for s in r.sources]) if r.sources else np.zeros((0, 2))
    truth = np.array([s[:2] for s in ev.sources])
    m = O.match(truth, est)
    primary = float(np.linalg.norm(est[0] - truth[0])) if len(est) else np.nan
    cover = np.nan
    d2 = det = np.nan
    vv = np.asarray(c.voltages_uv, float); amp = -vv[c.n_pre]
    nz = np.median(np.abs(vv[:20] - np.median(vv[:20], axis=0)), axis=0) / 0.6745
    live = np.ones(vv.shape[1], bool) if mask is None else np.asarray(mask, bool)
    snr = float(np.max(amp[live] / np.maximum(nz[live], 1e-6)))
    if r.sources:
        d = est[0] - truth[0]; C = np.asarray(r.sources[0].cov_xy)
        q = loc.cal.q_shared if "shared_waveform_pair" in r.flags else (loc.cal.q_two if r.n_sources >= 2 else (loc.cal.q_flagged if r.flags else loc.cal.q_clean))
        d2 = float(d @ np.linalg.solve(C, d)); det = float(np.linalg.det(C))
        cover = float(d2 <= q)
    return {"kind": ev.kind, "noise": ev.noise_uv, "dt": ev.dt, "sep": ev.sep_um, "res": ev.resolvable, "n_est": r.n_sources,
            "accepted": r.accepted, "flags": r.flags, "primary_err": primary, **m, "cover": cover,
            "area": (r.sources[0].region_area_um2 if r.sources else np.nan), "conf": r.confidence, "ms": ms, "primary_xy": (est[0].tolist() if len(est) else None), "d2": d2, "det": det, "snr": snr, "n_dead": int((~live).sum()), **info}
