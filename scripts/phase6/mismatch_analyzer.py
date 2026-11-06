"""Phase 6: classify every float-vs-fixed disagreement by its first divergent stage.

    .venv/bin/python scripts/phase6/mismatch_analyzer.py [--config v2|baseline] [--splits ...]

The float and integer detectors run on the same ADC codes. For every candidate or event present in
only one of them the analyzer finds the first stage at which the two pipelines disagree:

  crossing_flip:*   the below-threshold comparison differs at the crossing sample. The comparison margin changes
                    by e_filter + e_car_extra - e_baseline + e_offset; the term with the largest magnitude names the
                    source (filter, car, baseline, sigma_offset).
  refractory_cascade  crossing flags agree, but an earlier mismatch on the same channel moved last_peak
  peak_near_tie     crossing agrees, counterpart exists within +-3 samples, minima differ by less than the error at those samples (counterpart within the peak-search window)
  merge_amplitude_error_flip  candidate sets and merge group agree; the winner differs because the measured amplitude
                    deviation of the members (bounded by e_base, e_filter, e_car, sigma error, Q8.8 resolution) reverses the order
  candidate_cascade event differs because candidates in its merge neighbourhood differ
  UNEXPLAINED_*     none of the above: a potential implementation defect (must be investigated)

No algorithm is modified to improve agreement.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.phase5.compare_float_fixed import spec_from_float
from scripts.phase5.data import load_window, manifest_windows, window_path
from scripts.phase5.detectors import BASELINE, P5Config, filter_causal, run_detector
from scripts.phase5.fixed_detector import adc_quantize, run_fixed

V2 = BASELINE.replace(highpass_hz=200.0, lowpass_hz=4500.0)
CONFIGS = {"v2": V2, "baseline": BASELINE}


def dominant(terms: dict) -> str:
    return max(terms, key=lambda k: abs(terms[k]))


def analyse_window(w: dict, spec_w: dict, cfg: P5Config) -> dict:
    fspec = spec_from_float(cfg)
    ilsb = fspec.internal_lsb_uv
    codes = adc_quantize(w["voltages_uv"], fspec)
    vq = codes * (fspec.adc_lsb_uv_x32 / 32.0)
    fl = run_detector(vq, cfg, w["coordinates_um"])
    fx = run_fixed(codes, fspec)
    ff = filter_causal(vq, cfg)
    cal = fx["calibration"]
    S, R, M = fspec.peak_search, fspec.refractory, fspec.merge_window
    # error terms (microvolts)
    e_filter = fx["filtered"] * ilsb - ff
    e_car_extra = (fx["car"] * ilsb - fl["reference"]) - e_filter
    thr_f = fl["baseline"] - cfg.threshold_sigma * fl["sigma"]
    e_base = cal["baseline"] * ilsb - fl["baseline"]
    e_off = (cal["baseline"] - cal["threshold"]) * ilsb - cfg.threshold_sigma * fl["sigma"]
    below_f = fl["reference"] < thr_f[None, :]
    below_x = fx["car"] < cal["threshold"][None, :]
    ref_f, car_x = fl["reference"], fx["car"] * ilsb
    cf = {(e.sample, e.channel): e for e in fl["candidates"]}
    cx = {(e.sample, e.channel): e for e in fx["candidates"]}
    ef = {(e.sample, e.channel): e for e in fl["events"]}
    ex = {(e.sample, e.channel): e for e in fx["events"]}
    wid = {"session": spec_w["session"], "t0_s": spec_w["t0_s"], "split": spec_w["split"]}
    rows, cand_cat = [], {}

    def crossing_of(below, t, c):
        for i in range(t, max(0, t - S - 1), -1):
            if below[i, c] and not below[i - 1, c]:
                return i
        return None

    cross_f = below_f[1:] & ~below_f[:-1]; cross_x = below_x[1:] & ~below_x[:-1]  # index i -> crossing at sample i+1
    mism = [("float_only", k) for k in cf if k not in cx] + [("fixed_only", k) for k in cx if k not in cf]
    mism_by_ch = {}
    for side, (t, c) in mism:
        mism_by_ch.setdefault(c, []).append(t)
    for side, (t, c) in sorted(mism, key=lambda m: (m[1][0], m[1][1])):
        src_below = below_f if side == "float_only" else below_x
        lo_n, hi_n = max(1, t - (R + S + 2)), min(below_f.shape[0] - 1, t + S)
        dif = np.flatnonzero(cross_f[lo_n - 1:hi_n, c] != cross_x[lo_n - 1:hi_n, c]) + lo_n
        cat, detail = None, {}
        if len(dif):
            tc = int(dif[-1] if side == "fixed_only" and False else dif[0])
            terms = {"filter": e_filter[tc, c], "car": e_car_extra[tc, c], "baseline": -e_base[c], "sigma_offset": e_off[c]}
            cat = f"crossing_flip:{dominant(terms)}"
            detail = {"crossing_sample": tc, "margin_float_uv": float(ref_f[tc, c] - thr_f[c]),
                      "margin_fixed_uv": float(car_x[tc, c] - cal["threshold"][c] * ilsb),
                      "error_terms_uv": {k: float(v) for k, v in terms.items()}, "differing_crossings": [int(v) for v in dif[:4]]}
        if cat is None:
            prior = [u for u in mism_by_ch.get(c, []) if t - (R + S + 2) <= u < t]
            counterpart = [u for (s2, (u, c2)) in [(m[0], m[1]) for m in mism] if c2 == c and u != t and abs(u - t) <= S and ((side == "float_only") != (s2 == "float_only"))]
            if prior:
                cat, detail = "refractory_cascade", {"earlier_mismatch_samples": prior[:3]}
            elif counterpart:
                u = counterpart[0]
                tf, tx = (t, u) if side == "float_only" else (u, t)
                dgap = abs(float(ref_f[tf, c] - ref_f[tx, c]))
                bound = abs(e_filter[tf, c]) + abs(e_filter[tx, c]) + abs(e_car_extra[tf, c]) + abs(e_car_extra[tx, c])
                cat = "peak_near_tie" if dgap <= bound else "UNEXPLAINED_peak_shift"
                detail = {"counterpart_sample": int(u), "float_value_gap_uv": dgap, "error_bound_uv": float(bound)}
            else:
                cat = "UNEXPLAINED_no_flip_no_counterpart"
        cand_cat[(side, t, c)] = cat
        src_e = cf.get((t, c)) if side == "float_only" else cx.get((t, c))
        rows.append({**wid, "level": "candidate", "side": side, "channel": int(c), "sample": int(t),
                     "float_value_uv": float(src_e.voltage_uv) if side == "float_only" else None,
                     "fixed_value_uv": float(src_e.code * ilsb) if side == "fixed_only" else None,
                     "first_divergent_stage": cat, "difference_uv": detail.get("margin_float_uv", detail.get("float_value_gap_uv")),
                     "detail": detail})
    # events
    cand_f_set, cand_x_set = set(cf), set(cx)
    ev_rows = []
    from scripts.phase5.detectors import spatial_artifact_veto
    from scripts.phase5.fixed_detector import spatial_veto
    veto_f, _ = spatial_artifact_veto(fl['candidates'], w['voltages_uv'].shape[1], cfg)
    veto_x, _ = spatial_veto(fx['candidates'], w['voltages_uv'].shape[1], fspec)
    for side, (t, c) in [("float_only", k) for k in ef if k not in ex] + [("fixed_only", k) for k in ex if k not in ef]:
        lo, hi = t - 3 * M, t + 3 * M
        nf = {k for k in cand_f_set if lo <= k[0] <= hi}; nx = {k for k in cand_x_set if lo <= k[0] <= hi}
        e_src = ef[(t, c)] if side == "float_only" else ex[(t, c)]
        if nf != nx:
            diff = sorted(nf ^ nx)
            cats = sorted({cand_cat.get(("float_only" if k in nf else "fixed_only", k[0], k[1]), "?") for k in diff})
            cat, detail = "candidate_cascade", {"candidate_categories": cats, "differing_candidates": len(diff)}
        else:
            # candidate sets agree: rebuild the true merge group (post-veto, anchored on the first event) in both pipelines
            def group_of(events, merge_ms_samples):
                ordered = sorted(events, key=lambda e: (e.sample, e.channel)); groups = []
                for e in ordered:
                    if not groups or e.sample - groups[-1][0].sample > merge_ms_samples:
                        groups.append([e])
                    else:
                        groups[-1].append(e)
                for g in groups:
                    if any(e.sample == t and e.channel == c for e in g):
                        return g
                return None
            gf = group_of(veto_f, M); gx = group_of(veto_x, M)
            if gf is None or gx is None or {(e.sample, e.channel) for e in gf} != {(e.sample, e.channel) for e in gx}:
                cat, detail = "candidate_cascade", {"candidate_categories": ["veto_or_grouping_difference"]}
            else:
                over = []
                for e in gf:
                    xe = cx[(e.sample, e.channel)]
                    sig = fl["sigma"][e.channel]
                    r = abs(cal["sigma_q16"][e.channel] / 65536.0 * ilsb - sig) / sig
                    bound = (abs(e_base[e.channel]) + abs(e_filter[e.sample, e.channel]) + abs(e_car_extra[e.sample, e.channel])) / sig + e.normalized_amplitude * r + 2.0 / 256
                    dev = abs(xe.amp_q8 / 256.0 - e.normalized_amplitude)
                    if dev > bound:
                        over.append({"channel": e.channel, "sample": e.sample, "amp_deviation_sigma": dev, "bound_sigma": bound})
                amps = sorted([e.normalized_amplitude for e in gf], reverse=True)
                cat = "merge_amplitude_error_flip" if not over else "UNEXPLAINED_merge_decision"
                detail = {"float_amp_gap_sigma": (amps[0] - amps[1]) if len(amps) > 1 else None, "group": [(e.sample, e.channel, round(e.normalized_amplitude, 3), round(cx[(e.sample, e.channel)].amp_q8 / 256, 3)) for e in gf], "over_bound": over}
        ev_rows.append({**wid, "level": "event", "side": side, "channel": int(c), "sample": int(t),
                        "float_value_uv": float(e_src.voltage_uv) if side == "float_only" else None,
                        "fixed_value_uv": float(e_src.code * ilsb) if side == "fixed_only" else None,
                        "first_divergent_stage": cat, "difference_uv": None, "detail": detail})
    rows += ev_rows
    stats = {
        "float_candidates": len(cf), "fixed_candidates": len(cx), "candidates_exact": len(set(cf) & set(cx)),
        "float_events": len(ef), "fixed_events": len(ex), "events_exact": len(set(ef) & set(ex)),
        "filter_err_uv_rms": float(np.sqrt(np.mean(e_filter ** 2))), "filter_err_uv_max": float(np.abs(e_filter).max()),
        "car_extra_err_uv_max": float(np.abs(e_car_extra).max()), "baseline_err_uv_max": float(np.abs(e_base).max()),
        "offset_err_uv_max": float(np.abs(e_off).max()),
        "crossing_flag_disagreements": int(np.count_nonzero(below_f != below_x)),
    }
    return {"stats": stats, "rows": rows, "window": wid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v2", choices=list(CONFIGS))
    ap.add_argument("--splits", nargs="*", default=["calibration", "selection", "heldout", "seen"])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "phase6")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    a.out.mkdir(parents=True, exist_ok=True)
    all_rows, per_window = [], []
    tot = Counter()
    for split in a.splits:
        for spec in manifest_windows(split):
            p = window_path(spec["pid"], spec["t0_s"], spec["duration_s"])
            if not p.exists():
                continue
            r = analyse_window(load_window(p), spec, cfg)
            all_rows += r["rows"]
            per_window.append({**r["window"], **r["stats"]})
            for k in ("float_candidates", "fixed_candidates", "candidates_exact", "float_events", "fixed_events", "events_exact"):
                tot[k] += r["stats"][k]
            print(spec["session"], split, int(spec["t0_s"]), r["stats"]["float_events"], r["stats"]["events_exact"], flush=True)
    cc = Counter((r["level"], r["side"], r["first_divergent_stage"]) for r in all_rows)
    summary = {
        "config": a.config, "n_windows": len(per_window), "totals": dict(tot),
        "agreement_definitions": {
            "events_exact_over_float": tot["events_exact"] / max(1, tot["float_events"]),
            "events_jaccard": tot["events_exact"] / max(1, tot["float_events"] + tot["fixed_events"] - tot["events_exact"]),
            "candidates_exact_over_float": tot["candidates_exact"] / max(1, tot["float_candidates"]),
            "candidates_jaccard": tot["candidates_exact"] / max(1, tot["float_candidates"] + tot["fixed_candidates"] - tot["candidates_exact"]),
            "note": "98.4% in the Phase 5 report = events_exact_over_float: fraction of float events whose exact (sample, channel) is also a fixed event. Not tolerance based, not symmetric."},
        "categories": [{"level": k[0], "side": k[1], "first_divergent_stage": k[2], "count": v} for k, v in sorted(cc.items())],
        "unexplained": sum(v for k, v in cc.items() if k[2].startswith("UNEXPLAINED")),
    }
    (a.out / f"mismatch_summary_{a.config}.json").write_text(json.dumps({"summary": summary, "windows": per_window}, indent=1) + "\n")
    with (a.out / f"mismatches_{a.config}.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
