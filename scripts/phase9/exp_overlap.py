"""Phase 4/5 on real-like synthetic: two-source resolution + calibrated regions. Calibrate on dev, report select, final (once if --final)."""
import json, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, ".")
from edge.phase8.pipeline import Calibration, Phase8Localizer
from edge.phase9.improved import condition
from edge.phase9.synth_real import make_events, xy_error
OV_COND = dict(baseline=True, interp_dead=True)      # no CAR: common mode is a fitted nuisance (nuisance_cm=True)
P9_CONDITIONER = OV_COND

FINAL = "--final" in sys.argv
array, _ = make_events("dev", n_per_noise=1)
def build(pool, n):
    # singles + pairs + triples-free here (triples handled by Phase 8 benchmark); 50% pairs
    return make_events(pool, n_per_noise=n, contaminated_fraction=0.5)[1]

def loc_for(cal=None, rank=4):
    return Phase8Localizer(array, cal, model="line_source", lambda_um=20.0, rank=rank, max_x_um=90.0, y_margin_um=40.0, nuisance_cm=True)

def analyze(loc, evs):
    return [loc.analyze(condition(e.snippet, **P9_CONDITIONER)) for e in evs]

def calibrate(evs, false_two=0.03, false_sh=0.01, accept=0.92):
    loc = loc_for(); cal = Calibration(); an = analyze(loc, evs)
    single = np.array([not e.contaminated for e in evs])
    d = np.array([a[0].delta for a in an]); ds = np.array([a[0].delta_shared for a in an])
    cal.tau_two = float(np.percentile(d[single], 100 * (1 - false_two))); cal.tau_gray = float(np.percentile(d[single], 90))
    cal.tau_shared = float(np.percentile(ds[single], 100 * (1 - false_sh)))
    cal.max_chi2_red = 1e9; loc.cal = cal
    chis = [loc.localize(e.snippet, a).chi2_red for e, a, s in zip(evs, an, single) if s]
    cal.max_chi2_red = float(1.1 * np.percentile(chis, 98)); loc.cal = cal
    areas = []
    for e, a in zip(evs, an):
        r = loc.localize(e.snippet, a)
        if r.primary is not None and "chi2_red" not in r.reject_reason:
            areas.append(float(np.pi * np.sqrt(max(np.linalg.det(np.array(r.primary.cov_xy)), 0))))
    cal.max_unit_area_um2 = float(np.percentile(areas, 100 * accept)); loc.cal = cal
    sc = {"clean": [], "flagged": [], "two": [], "shared": []}
    for e, a in zip(evs, an):
        r = loc.localize(e.snippet, a)
        if r.primary is None or not r.accepted: continue
        d_ = np.array(r.primary.xyz_um[:2]) - np.array(e.truth_xy)
        m = float(d_ @ np.linalg.solve(np.array(r.primary.cov_xy), d_))
        key = ("shared" if "shared_waveform_pair" in r.flags else "two") if r.n_sources == 2 else ("flagged" if r.flags else "clean")
        sc[key].append(m)
    def q(v):
        if len(v) < 10: return 20.0
        return float(np.sort(v)[min(int(np.ceil((len(v) + 1) * 0.9)), len(v)) - 1])
    cal.q_clean, cal.q_flagged, cal.q_two, cal.q_shared = q(sc["clean"]), q(sc["flagged"]), q(sc["two"]), q(sc["shared"])
    return cal

def evaluate(cal, evs, base_engine):
    loc = loc_for(cal); rows = []
    for e in evs:
        sn = condition(e.snippet, **P9_CONDITIONER)
        r = loc.localize(sn)
        truth = np.array([t[:2] for t in e.sources_full[:]] ) if False else np.array(e.sources_xy) if hasattr(e, "sources_xy") else np.array(e.truth_xy)[None]
        tru = np.array([(s[0], s[1]) for s in e.sources_full])
        est = np.array([s.xyz_um[:2] for s in r.sources]) if r.sources else np.zeros((0, 2))
        if len(est):
            D = np.linalg.norm(tru[:, None] - est[None], axis=2); ri, ci = linear_sum_assignment(D)
            ok = D[ri, ci] <= 60; matched = int(ok.sum()); merr = D[ri, ci][ok].tolist()
            ev_err = float(np.mean([D[t].min() if t not in ri[ok] else D[t, ci[list(ri).index(t)]] for t in range(len(tru))]))
        else:
            matched, merr, ev_err = 0, [], float("nan")
        # P9 single-source engine on the same event (one source always)
        b = base_engine.estimate(e.snippet)     # P9 engine conditions internally
        b_err = xy_error(b.xyz_um, e.truth_xy) if b.available else np.nan
        cov = cover = area = np.nan
        if r.primary is not None:
            d_ = np.array(r.primary.xyz_um[:2]) - np.array(e.truth_xy)
            C = np.array(r.primary.cov_xy)
            q = cal.q_shared if "shared_waveform_pair" in r.flags else (cal.q_two if r.n_sources == 2 else (cal.q_flagged if r.flags else cal.q_clean))
            cover = float(d_ @ np.linalg.solve(C, d_) <= q); area = r.primary.region_area_um2
        rows.append(dict(kind=e.kind, res=e.resolvable, noise=e.noise_uv, n_est=r.n_sources, accepted=r.accepted, matched=matched, merr=merr,
                         ev_err=ev_err, primary_err=(float(np.linalg.norm(np.array(r.primary.xyz_um[:2]) - np.array(e.truth_xy))) if r.primary else np.nan),
                         base_err=b_err, cover=cover, area=area, unit_area=(float(np.pi*np.sqrt(max(np.linalg.det(np.array(r.primary.cov_xy)),0))) if r.primary else np.nan),
                         flags=r.flags, reason=r.reject_reason, ms=r.runtime_ms))
    return rows

def summarize(rows):
    A = lambda f, sel=lambda r: True: [f(r) for r in rows if sel(r)]
    sing = [r for r in rows if r["kind"] == "single"]; pairs = [r for r in rows if r["kind"] == "pair"]; res = [r for r in rows if r["res"]]
    me = [x for r in res if r["n_est"] == 2 and r["matched"] == 2 for x in r["merr"]]
    acc = [r for r in rows if r["accepted"]]
    pe = np.array([r["primary_err"] for r in acc]); be = np.array([r["base_err"] for r in rows if np.isfinite(r["base_err"])])
    pe_all = np.array([r["primary_err"] for r in rows]); 
    return {"n": len(rows), "singles": len(sing), "pairs": len(pairs), "resolvable_pairs": len(res),
            "two_source_recall_resolvable": float(np.mean([r["n_est"] >= 2 and r["matched"] == 2 and r["accepted"] for r in res])),
            "two_source_recall_all_pairs": float(np.mean([r["n_est"] >= 2 and r["matched"] == 2 and r["accepted"] for r in pairs])),
            "false_two_source_rate_singles": float(np.mean([r["n_est"] >= 2 for r in sing])),
            "per_source_median_err_correct_pairs_um": float(np.median(me)) if me else None, "n_correct_pairs": int(len(me) // 2),
            "rejected_all": float(np.mean([not r["accepted"] for r in rows])), "rejected_resolvable": float(np.mean([not r["accepted"] for r in res])),
            "p9_single_engine_pair_event_err_p90": float(np.nanpercentile([r["base_err"] for r in pairs], 90)),
            "k2_pair_event_err_p90": float(np.nanpercentile([r["ev_err"] for r in pairs if np.isfinite(r["ev_err"])], 90)),
            "p9_single_engine_resolvable_err_median": float(np.nanmedian([r["base_err"] for r in res])),
            "k2_resolvable_event_err_median": float(np.nanmedian([r["ev_err"] for r in res])),
            "accepted_fraction": float(len(acc) / len(rows)),
            "accepted_primary_err": {"median": float(np.nanmedian(pe)), "p90": float(np.nanpercentile(pe, 90)), "p95": float(np.nanpercentile(pe, 95))},
            "large_error_rate_primary_accepted": float(np.mean(pe > 50)), "large_error_rate_single_engine": float(np.mean(be > 50)),
            "region_coverage_accepted": float(np.nanmean([r["cover"] for r in acc])),
            "region_coverage_singles": float(np.nanmean([r["cover"] for r in acc if r["kind"] == "single"])),
            "region_coverage_pairs": float(np.nanmean([r["cover"] for r in acc if r["kind"] == "pair"])),
            "region_area_um2_median": float(np.nanmedian([r["area"] for r in acc])), "runtime_ms_median": float(np.median([r["ms"] for r in rows]))}

from edge.phase9.configs import p9_factory
base = p9_factory(array)["weighted_scan"]    # single-source comparison engine (full P9 conditioning incl. CAR)
from edge.phase9.improved import ConditionedLocalizer
t = time.time(); dev = build("dev", 120); cal = calibrate(dev); print("calibrated", round(time.time()-t), "s", {k: round(v, 2) for k, v in cal.to_json().items()}, flush=True)
out = {"calibration": cal.to_json()}
sel = build("select", 100); out["select"] = summarize(evaluate(cal, sel, base)); print("select", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["select"].items()}), flush=True)
if FINAL:
    fin = build("final", 150); rows = evaluate(cal, fin, base); out["final"] = summarize(rows)
    # risk-coverage by uncertainty
    ok = [r for r in rows if np.isfinite(r["primary_err"])]; order = sorted(ok, key=lambda r: r["unit_area"])
    out["risk_coverage"] = [{"kept": f, "median": float(np.median([r["primary_err"] for r in order[:int(f*len(order))]])), "p90": float(np.percentile([r["primary_err"] for r in order[:int(f*len(order))]], 90)),
                             "large_rate": float(np.mean([r["primary_err"] > 50 for r in order[:int(f*len(order))]]))} for f in (0.5, 0.7, 0.9, 1.0)]
    out["by_noise"] = {str(n): summarize([r for r in rows if r["noise"] == n]) for n in (5.0, 10.0, 20.0)}
    print("final", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["final"].items()}))
    json.dump(out, open("reports/phase9/overlap_uncertainty_final.json", "w"), indent=1, default=float)
else:
    json.dump(out, open("reports/phase9/overlap_uncertainty_select.json", "w"), indent=1, default=float)
