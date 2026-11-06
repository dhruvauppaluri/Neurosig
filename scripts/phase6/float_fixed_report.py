"""Float vs fixed comparison on every cached window, reporting two separate questions:

  A. numerical equivalence: how closely does the integer model reproduce the float algorithm?
  B. detection quality: does the integer model keep agreement with the independent sorter reference (chance-adjusted)?

    .venv/bin/python scripts/phase6/float_fixed_report.py --config v2 [--splits ...]
High float/fixed agreement is not evidence of biological detection correctness (sorter output is not ground truth).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.phase5.benchmark import RADIUS_UM, NULL_SHIFTS_S
from scripts.phase5.compare_float_fixed import spec_from_float
from scripts.phase5.data import load_window, window_path
from scripts.phase6.manifests import specs as manifest_specs
from scripts.phase5.detectors import run_detector
from scripts.phase5.fixed_detector import adc_quantize, run_fixed
from scripts.phase5.metrics import match_optimal, paired_bootstrap_diff, pooled, window_metrics
from scripts.phase6.mismatch_analyzer import CONFIGS


def quality(events, w):
    fs, dur = w["fs"], w["duration_s"]
    m = window_metrics(events, w["sorter_times_s"], fs, dur, 1.0, w["sorter_local_channel"], w["coordinates_um"], RADIUS_UM)
    tps = []
    for sh in NULL_SHIFTS_S:
        s = (w["sorter_times_s"] + sh) % dur; o = np.argsort(s)
        tps.append(window_metrics(events, s[o], fs, dur, 1.0, w["sorter_local_channel"][o], w["coordinates_um"], RADIUS_UM)["tp"])
    return {k: m[k] for k in ("tp", "fp", "fn", "n_detected", "n_reference", "precision", "recall", "f1")} | {"chance_tp": float(np.mean(tps))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v2", choices=list(CONFIGS))
    ap.add_argument("--car", default=None, choices=[None, "mean"])
    ap.add_argument("--splits", nargs="*", default=["calibration", "selection", "heldout", "seen"], help="splits of the chosen manifest")
    ap.add_argument("--manifest", default="phase5", choices=["phase5", "phase6"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    if a.car:
        cfg = cfg.replace(car=a.car)
    fspec = spec_from_float(cfg); ilsb = fspec.internal_lsb_uv; lsb = fspec.adc_lsb_uv_x32 / 32.0
    rows = []
    for split in a.splits:
        for spec in manifest_specs(split, a.manifest):
            p = window_path(spec["pid"], spec["t0_s"], spec["duration_s"])
            if not p.exists():
                continue
            w = load_window(p)
            codes = adc_quantize(w["voltages_uv"], fspec)
            fl = run_detector(codes * lsb, cfg, w["coordinates_um"]); fx = run_fixed(codes, fspec)
            kf = {(e.sample, e.channel): e for e in fl["events"]}; kx = {(e.sample, e.channel): e for e in fx["events"]}
            exact = set(kf) & set(kx)
            fs_ = np.array([e.sample for e in fl["events"]]); xs_ = np.array([e.sample for e in fx["events"]])
            tol = {}
            for t in (1, 3, 15):
                tol[str(t)] = len(match_optimal(xs_, fs_, t))   # sample tolerance, any channel
            tol_ch = {}
            for t in (3, 15):  # tolerance AND same channel
                allowed = (np.array([e.channel for e in fx["events"]])[:, None] == np.array([e.channel for e in fl["events"]])[None, :]) if len(fs_) and len(xs_) else None
                tol_ch[str(t)] = len(match_optimal(xs_, fs_, t, allowed)) if allowed is not None else 0
            amp = [abs(kx[k].amp_q8 / 256.0 - kf[k].normalized_amplitude) for k in exact]
            code = [abs(kx[k].code * ilsb - kf[k].voltage_uv) for k in exact]
            rows.append(dict(session=spec["session"], split=spec["split"], t0_s=spec["t0_s"], duration_s=w["duration_s"],
                             float_events=len(kf), fixed_events=len(kx), exact=len(exact), float_only=len(set(kf) - set(kx)), fixed_only=len(set(kx) - set(kf)),
                             tolerance_matches_any_channel=tol, tolerance_matches_same_channel=tol_ch,
                             amp_abs_diff_sigma=amp, code_abs_diff_uv=code,
                             float_candidates=len(fl["candidates"]), fixed_candidates=len(fx["candidates"]),
                             quality_float=quality(fl["events"], w), quality_fixed=quality(fx["events"], w)))
            print(spec["session"], spec["split"], int(spec["t0_s"]), len(kf), len(exact), flush=True)

    def agg(sub):
        s = lambda k: sum(r[k] for r in sub)
        amp = np.concatenate([r["amp_abs_diff_sigma"] for r in sub]) if sub else np.array([0.0]); cd = np.concatenate([r["code_abs_diff_uv"] for r in sub]) if sub else np.array([0.0])
        out = dict(windows=len(sub), duration_s=s("duration_s"), float_events=s("float_events"), fixed_events=s("fixed_events"), exact_matches=s("exact"),
                   float_only=s("float_only"), fixed_only=s("fixed_only"),
                   exact_over_float=s("exact") / max(1, s("float_events")), jaccard=s("exact") / max(1, s("float_events") + s("fixed_events") - s("exact")),
                   tol_any_channel={t: sum(r["tolerance_matches_any_channel"][t] for r in sub) for t in ("1", "3", "15")},
                   tol_same_channel={t: sum(r["tolerance_matches_same_channel"][t] for r in sub) for t in ("3", "15")},
                   unmatched_float_at_15=s("float_events") - sum(r["tolerance_matches_same_channel"]["15"] for r in sub),
                   amp_diff_sigma_exact_events={"median": float(np.median(amp)), "p99": float(np.percentile(amp, 99)), "max": float(amp.max())},
                   code_diff_uv_exact_events={"median": float(np.median(cd)), "p99": float(np.percentile(cd, 99)), "max": float(cd.max())})
        for side in ("float", "fixed"):
            q = [r[f"quality_{side}"] for r in sub]; p = pooled(q)
            tp_c = sum(x["chance_tp"] for x in q); nd, nr = p["n_detected"], p["n_reference"]
            cr, cp = tp_c / max(1, nr), tp_c / max(1, nd)
            out[f"sorter_{side}"] = dict(f1=p["f1"], recall=p["recall"], precision=p["precision"], chance_f1=(2 * cp * cr / (cp + cr) if cp + cr else 0.0),
                                         excess_f1=p["f1"] - (2 * cp * cr / (cp + cr) if cp + cr else 0.0), n_detected=nd, n_reference=nr)
        pd = paired_bootstrap_diff([r["quality_fixed"] for r in sub], [r["quality_float"] for r in sub], "f1", seed=3)
        out["fixed_minus_float_f1"] = {"diff": pd["diff"], "ci95": pd["ci"]}
        return out
    summary = {"config": a.config + ("+mean" if a.car else ""), "all": agg(rows)}
    summary["by_session"] = {s: agg([r for r in rows if r["session"] == s]) for s in sorted({r["session"] for r in rows})}
    summary["by_split"] = {s: agg([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})}
    out = a.out or ROOT / "reports" / "phase6" / f"float_vs_fixed_{summary['config'].replace('+', '_')}.json"
    slim = [{k: v for k, v in r.items() if k not in ("amp_abs_diff_sigma", "code_abs_diff_uv")} for r in rows]
    out.write_text(json.dumps({"summary": summary, "windows": slim}, indent=1) + "\n")
    print(json.dumps(summary["all"], indent=1))


if __name__ == "__main__":
    main()
