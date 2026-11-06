"""Multi-window, multi-session detector benchmark against the sorter reference.

    .venv/bin/python scripts/phase5/benchmark.py --split selection --variants singles
    .venv/bin/python scripts/phase5/benchmark.py --split heldout --configs reports/phase5/detector-v2.json

Primary metric: channel-consistent F1 at 1 ms (see reports/phase5/selection_criteria.json).
Every metric is also reported with the shifted-reference chance level. Sorting is not
ground truth, so "precision" below is agreement rate, not biological precision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.phase5.data import load_window, manifest_windows, window_path
from scripts.phase5.detectors import BASELINE, P5Config, filter_causal, run_detector
from scripts.phase5.metrics import TOLERANCES_MS, bootstrap_ci, paired_bootstrap_diff, pooled, timing_summary, window_metrics

RADIUS_UM = 60.0
NULL_SHIFTS_S = (0.041, 0.073)
OUT = ROOT / "reports" / "phase5"


def evaluate_window(w: dict, spec: dict, cfg: P5Config, filt_cache: dict) -> dict:
    fs, dur = w["fs"], w["duration_s"]
    key = (cfg.highpass_hz, cfg.lowpass_hz)
    if key not in filt_cache:
        filt_cache[key] = filter_causal(w["voltages_uv"], cfg)
    run = run_detector(w["voltages_uv"], cfg, w["coordinates_um"], filtered_cache=filt_cache[key])
    ev = run["events"]
    ref_t = w["sorter_times_s"]
    ref_ch = w["sorter_local_channel"]
    out = {"session": spec["session"], "t0_s": spec["t0_s"], "split": spec["split"],
           "n_events": len(ev), "n_candidates": run["n_candidates"], "runtime_s": run["runtime_s"],
           "sigma_uv_median": float(np.median(run["sigma"])), "tol": {}}
    for tol in TOLERANCES_MS:
        unc = window_metrics(ev, ref_t, fs, dur, tol)
        chn = window_metrics(ev, ref_t, fs, dur, tol, ref_ch, w["coordinates_um"], RADIUS_UM)
        nulls_unc, nulls_chn = [], []
        for sh in NULL_SHIFTS_S:
            shifted = (ref_t + sh) % dur
            order = np.argsort(shifted)
            nulls_unc.append(window_metrics(ev, shifted[order], fs, dur, tol))
            nulls_chn.append(window_metrics(ev, shifted[order], fs, dur, tol, ref_ch[order], w["coordinates_um"], RADIUS_UM))
        slim = lambda m: {k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "n_detected", "n_reference", "false_detections_per_s")}
        mean_tp = lambda ms: float(np.mean([m["tp"] for m in ms]))
        out["tol"][str(tol)] = {
            "unconstrained": slim(unc), "channel_consistent": slim(chn),
            "chance_unconstrained_tp": mean_tp(nulls_unc), "chance_channel_consistent_tp": mean_tp(nulls_chn),
            "timing_error_ms_channel_consistent": chn["timing_error_ms"],
        }
        if tol == 1.0:
            # per-reference-channel recall (channel-consistent)
            hit = np.zeros(len(ref_t), dtype=bool)
            hit[chn["matched_ref_idx"]] = True
            out["per_channel_recall_1ms"] = {int(c): [int(hit[ref_ch == c].sum()), int((ref_ch == c).sum())]
                                             for c in np.unique(ref_ch)}
    return out


def load_split(split: str) -> list[tuple[dict, dict]]:
    items = []
    for spec in manifest_windows(split):
        p = window_path(spec["pid"], spec["t0_s"], spec["duration_s"])
        if not p.exists():
            raise FileNotFoundError(f"missing window {p.name}; run scripts/phase5/fetch_all.py")
        w = load_window(p)
        if w["sync_version"] < 3:
            raise RuntimeError(f"{p.name} has not been resynced to the probe clock")
        items.append((w, spec))
    return items


def run_config(items, cfg: P5Config, caches: list[dict]) -> list[dict]:
    return [evaluate_window(w, spec, cfg, caches[i]) for i, (w, spec) in enumerate(items)]


def per_tol_windows(results: list[dict], tol: float, mode: str) -> list[dict]:
    return [r["tol"][str(tol)][mode] for r in results]


def summarize(results: list[dict], seed: int = 0) -> dict:
    s = {"n_windows": len(results), "n_events": sum(r["n_events"] for r in results),
         "runtime_s_total": sum(r["runtime_s"] for r in results),
         "samples_total": None, "tol": {}, "per_session": {}}
    dur = sum(2.0 if r["split"] != "seen" else 1.0 for r in results)
    s["duration_s_total"] = dur
    for tol in TOLERANCES_MS:
        t = {}
        for mode in ("unconstrained", "channel_consistent"):
            wins = per_tol_windows(results, tol, mode)
            p = pooled(wins)
            ci_f1 = bootstrap_ci(wins, "f1", seed=seed)
            ci_rec = bootstrap_ci(wins, "recall", seed=seed)
            chance_key = "chance_unconstrained_tp" if mode == "unconstrained" else "chance_channel_consistent_tp"
            chance_tp = sum(r["tol"][str(tol)][chance_key] for r in results)
            nd, nr = p["n_detected"], p["n_reference"]
            c_rec = chance_tp / nr if nr else 0.0
            c_prec = chance_tp / nd if nd else 0.0
            c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if c_prec + c_rec else 0.0
            t[mode] = {**{k: p[k] for k in ("tp", "fp", "fn", "n_detected", "n_reference", "precision", "recall", "f1", "macro_recall", "macro_f1")},
                       "f1_ci95": ci_f1, "recall_ci95": ci_rec,
                       "chance_f1": c_f1, "chance_recall": c_rec, "chance_precision": c_prec,
                       "false_detections_per_s": p["fp"] / dur}
        errs = [e for r in results for e in r["tol"][str(tol)]["timing_error_ms_channel_consistent"]]
        t["timing_channel_consistent"] = timing_summary(errs)
        s["tol"][str(tol)] = t
    for sess in sorted({r["session"] for r in results}):
        sub = [r for r in results if r["session"] == sess]
        s["per_session"][sess] = {
            str(tol): {m: {k: pooled(per_tol_windows(sub, tol, m))[k] for k in ("recall", "precision", "f1", "n_reference")}
                       for m in ("unconstrained", "channel_consistent")} for tol in TOLERANCES_MS}
    return s


def compare(base: list[dict], other: list[dict], tol: float = 1.0, mode: str = "channel_consistent") -> dict:
    a, b = per_tol_windows(other, tol, mode), per_tol_windows(base, tol, mode)
    d = paired_bootstrap_diff(a, b, "f1", seed=1)
    sessions = {}
    for sess in sorted({r["session"] for r in base}):
        ia = [i for i, r in enumerate(base) if r["session"] == sess]
        sessions[sess] = pooled([a[i] for i in ia])["f1"] - pooled([b[i] for i in ia])["f1"]
    fa = pooled(a)["fp"] / max(1e-9, sum(2.0 for _ in a)); fb = pooled(b)["fp"] / max(1e-9, sum(2.0 for _ in b))
    rt = sum(r["runtime_s"] for r in other) / max(1e-9, sum(r["runtime_s"] for r in base))
    return {"delta_f1": d["diff"], "delta_f1_ci95": d["ci"], "fraction_windows_better": d["fraction_windows_a_better"],
            "delta_f1_per_session": sessions, "fdr_ratio": fa / fb if fb else None, "runtime_ratio": rt,
            "delta_recall": pooled(a)["recall"] - pooled(b)["recall"],
            "delta_precision": pooled(a)["precision"] - pooled(b)["precision"]}


def adopt(c: dict) -> tuple[bool, list[str]]:
    reasons = []
    if c["delta_f1_ci95"][0] <= 0:
        reasons.append("CI includes 0")
    if c["delta_f1"] < 0.01:
        reasons.append("delta F1 < 0.01")
    if min(c["delta_f1_per_session"].values()) < -0.02:
        reasons.append("a session regressed > 0.02")
    if c["fdr_ratio"] is not None and c["fdr_ratio"] > 1.25:
        reasons.append("false detections/s up > 25%")
    if c["runtime_ratio"] > 3.0:
        reasons.append("runtime > 3x")
    return (not reasons), reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["calibration", "selection", "heldout"])
    ap.add_argument("--variants", default="singles", help="name of a variant set in variants.py")
    ap.add_argument("--configs", type=Path, help="JSON {name: P5Config dict} overriding --variants")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.split == "heldout" and not args.configs:
        raise SystemExit("heldout may only be run with an explicit frozen --configs file")
    if args.configs:
        raw = json.loads(args.configs.read_text())
        raw = raw.get("configs", raw)
        variants = {k: P5Config(**v) for k, v in raw.items()}
    else:
        from scripts.phase5.variants import VARIANT_SETS
        variants = VARIANT_SETS[args.variants]
    items = load_split(args.split)
    caches = [dict() for _ in items]
    out = {"split": args.split, "n_windows": len(items), "tolerances_ms": list(TOLERANCES_MS),
           "channel_radius_um": RADIUS_UM, "null_shifts_s": list(NULL_SHIFTS_S), "variants": {}}
    base_res = None
    for name, cfg in variants.items():
        t = time.time()
        res = run_config(items, cfg, caches)
        summ = summarize(res)
        entry = {"config": cfg.as_dict(), "summary": summ, "per_window": [
            {k: v for k, v in r.items() if k != "tol"} | {"tol": {t_: {m: v for m, v in d.items() if m != "timing_error_ms_channel_consistent"} for t_, d in r["tol"].items()}} for r in res]}
        if name == "baseline":
            base_res = res
        if base_res is not None and name != "baseline":
            entry["vs_baseline_1ms_channel_consistent"] = compare(base_res, res)
            ok, why = adopt(entry["vs_baseline_1ms_channel_consistent"])
            entry["adopt_rule_passed"] = ok
            entry["adopt_rule_failures"] = why
        out["variants"][name] = entry
        f = summ["tol"]["1.0"]["channel_consistent"]
        print(f"{name:34s} ev={summ['n_events']:5d} chF1={f['f1']:.3f} rec={f['recall']:.3f} prec={f['precision']:.3f} "
              f"chance_f1={f['chance_f1']:.3f} fdr/s={f['false_detections_per_s']:.1f} {time.time()-t:.0f}s", flush=True)
    path = args.out or OUT / f"benchmark_{args.split}_{args.variants if not args.configs else args.configs.stem}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
