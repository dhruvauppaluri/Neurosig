"""Stage-by-stage comparison of the float detector and the integer reference.

    .venv/bin/python scripts/phase5/compare_float_fixed.py --split calibration \
        --float-config reports/phase5/detector-v2.json --out reports/phase5/float_vs_fixed_calibration.json

Both run on the *same* ADC codes (float input = codes * ADC LSB). This does not
claim bit equality between float and fixed; it quantifies the differences and
attributes them to stages. The integer model is the specification for RTL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.phase5.benchmark import RADIUS_UM, load_split
from scripts.phase5.detectors import BASELINE, P5Config, run_detector
from scripts.phase5.fixed_detector import FixedSpec, adc_quantize, run_fixed
from scripts.phase5.metrics import match_optimal, window_metrics


def spec_from_float(cfg: P5Config, base: FixedSpec | None = None) -> FixedSpec:
    """Map the float configuration onto the integer spec (same algorithmic settings)."""
    base = base or FixedSpec()
    fs = int(round(cfg.sample_rate_hz))
    ms = lambda v: int(round(v * fs / 1000.0))
    return FixedSpec(
        sample_rate_hz=fs, highpass_hz=cfg.highpass_hz, lowpass_hz=float(cfg.lowpass_hz),
        car=cfg.car if cfg.car in ("median", "mean", "none") else "median",
        threshold_mult_q8=int(round(cfg.threshold_sigma * 256)),
        mad_iterations=cfg.mad_iterations, peak_search=ms(cfg.peak_search_ms), refractory=ms(cfg.refractory_ms),
        merge_window=ms(cfg.merge_ms), veto_window=ms(0.2),
        veto_fraction_q8=int(round(cfg.artifact_channel_fraction * 256)) if cfg.artifact_channel_fraction <= 1.0 else 0,
        guard_bits=base.guard_bits, adc_lsb_uv_x32=base.adc_lsb_uv_x32)


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    return {"max_abs": float(np.max(np.abs(a))), "rms": float(np.sqrt(np.mean(a ** 2))), "mean": float(np.mean(a))}


def compare_window(w: dict, cfg: P5Config, spec: FixedSpec | None = None) -> dict:
    spec = spec or spec_from_float(cfg)
    lsb = spec.adc_lsb_uv_x32 / 32.0
    codes = adc_quantize(w["voltages_uv"], spec)
    v_q = codes * lsb  # what the float path sees after the ADC
    fs = w["fs"]
    fl = run_detector(v_q, cfg, w["coordinates_um"])
    fx = run_fixed(codes, spec)
    ilsb = spec.internal_lsb_uv
    out = {"adc_quantization_error_uv": _stats(w["voltages_uv"] - v_q)}
    # filter
    from scripts.phase5.detectors import filter_causal
    ff = filter_causal(v_q, cfg)
    out["filter_uv"] = _stats(fx["filtered"] * ilsb - ff)
    sigma_f = fl["sigma"]
    out["filter_error_over_sigma_rms"] = float(np.sqrt(np.mean(((fx["filtered"] * ilsb - ff) / sigma_f) ** 2)))
    out["filter_stats_fixed"] = [{k: s[k] for k in ("saturated_samples", "accumulator_bits_required_including_sign")}
                                 for s in fx["filter_stats"]]
    # CAR
    out["car_uv"] = _stats(fx["car"] * ilsb - fl["reference"])
    # calibration / threshold
    cal = fx["calibration"]
    thr_f = fl["baseline"] - cfg.threshold_sigma * fl["sigma"]
    out["baseline_uv"] = _stats(cal["baseline"] * ilsb - fl["baseline"])
    out["sigma_rel_error"] = _stats((cal["sigma_q16"] / 65536.0 * ilsb - fl["sigma"]) / fl["sigma"])
    out["threshold_uv"] = _stats(cal["threshold"] * ilsb - thr_f)
    # candidates / events
    def key(es): return {(e.sample, e.channel) for e in es}
    cf, cx = key(fl["candidates"]), key(fx["candidates"])
    ef, ex = key(fl["events"]), key(fx["events"])
    out["candidates"] = {"float": len(cf), "fixed": len(cx), "identical": len(cf & cx),
                         "float_only": len(cf - cx), "fixed_only": len(cx - cf)}
    # peak timing among candidates matched by channel within 3 samples
    dts = []
    for ch in range(w["voltages_uv"].shape[1]):
        a = np.array(sorted(e.sample for e in fl["candidates"] if e.channel == ch))
        b = np.array(sorted(e.sample for e in fx["candidates"] if e.channel == ch))
        for i, j, dt in match_optimal(b, a, 3):
            dts.append(dt)
    out["candidate_peak_offset_samples_fixed_minus_float"] = {
        "n": len(dts), "nonzero": int(np.count_nonzero(dts)), "max_abs": int(np.max(np.abs(dts))) if dts else 0}
    out["events"] = {"float": len(ef), "fixed": len(ex), "identical": len(ef & ex),
                     "float_only": len(ef - ex), "fixed_only": len(ex - ef),
                     "vetoed_float": fl["n_artifact_vetoed"], "vetoed_fixed": fx["n_vetoed"]}
    ed = []
    pairs = match_optimal(np.array([e.sample for e in fx["events"]]), np.array([e.sample for e in fl["events"]]), 15)
    ed = [p[2] for p in pairs]
    out["event_timing_offset_samples_fixed_minus_float"] = {"n": len(ed), "median": float(np.median(ed)) if ed else None,
                                                            "max_abs": int(np.max(np.abs(ed))) if ed else 0}
    # amplitude of identical events
    fa = {(e.sample, e.channel): e for e in fl["events"]}
    rel = []
    for e in fx["events"]:
        k = (e.sample, e.channel)
        if k in fa:
            rel.append((e.amp_q8 / 256.0) - fa[k].normalized_amplitude)
    out["event_amplitude_sigma_error"] = _stats(rel) if rel else None
    # agreement with the sorter reference
    ref_t, ref_ch = w["sorter_times_s"], w["sorter_local_channel"]
    for name, evs in (("float", fl["events"]), ("fixed", fx["events"])):
        m = window_metrics(evs, ref_t, fs, w["duration_s"], 1.0, ref_ch, w["coordinates_um"], RADIUS_UM)
        out[f"sorter_channel_consistent_1ms_{name}"] = {k: m[k] for k in ("tp", "fp", "fn", "f1", "precision", "recall")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["calibration", "selection", "heldout"])
    ap.add_argument("--float-config", type=Path, help="JSON with 'configs'/'name' or a bare config; default baseline")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = BASELINE
    if args.float_config:
        raw = json.loads(args.float_config.read_text())
        raw = raw.get("configs", raw)
        cfg = P5Config(**(raw[args.name] if args.name in raw else raw))
    spec = spec_from_float(cfg)
    res = []
    for w, spec_w in load_split(args.split):
        r = compare_window(w, cfg, spec)
        r.update(session=spec_w["session"], t0_s=spec_w["t0_s"])
        res.append(r)
        print(spec_w["session"], int(spec_w["t0_s"]), "events float/fixed/identical",
              r["events"]["float"], r["events"]["fixed"], r["events"]["identical"], flush=True)
    tot = {k: sum(r["events"][k] for r in res) for k in ("float", "fixed", "identical", "float_only", "fixed_only")}
    ctot = {k: sum(r["candidates"][k] for r in res) for k in ("float", "fixed", "identical", "float_only", "fixed_only")}
    args.out.write_text(json.dumps({"split": args.split, "float_config": cfg.as_dict(), "fixed_spec": spec.as_dict(),
                                    "totals_events": tot, "totals_candidates": ctot, "windows": res}, indent=1) + "\n")
    print("events", tot, "candidates", ctot)


if __name__ == "__main__":
    main()
