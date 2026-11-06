"""Reproducible error budget for float-vs-fixed differences (detector-v2.0 band, fixed-v1.1).

Separates coefficient quantisation, truncation arithmetic, calibration arithmetic and the CAR stage, measures the gain shift and
noise-estimate bias they cause, compares rounding rules (diagnostic only; the frozen rule is truncation toward zero), tests the
zero-input limit cycle, and evaluates the analytic worst-case truncation bound.
    .venv/bin/python scripts/phase6/error_decomposition.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import signal

from scripts.phase5.compare_float_fixed import spec_from_float
from scripts.phase5.data import load_window, manifest_windows, window_path
from scripts.phase5.detectors import common_reference, estimate_noise_variant, filter_causal, run_detector
from scripts.phase5.fixed_detector import adc_quantize, run_fixed
from scripts.phase6.mismatch_analyzer import V2

fs = spec_from_float(V2); ilsb = fs.internal_lsb_uv
rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
gain = lambda a, b: float((a * b).sum() / (b * b).sum() - 1)
specs = manifest_windows("heldout")[:8]
agg = {k: [] for k in ("filter_total_rms_uv", "filter_coef_effect_rms_uv", "filter_trunc_effect_rms_uv", "filter_max_abs_uv", "gain_int_vs_float_q", "gain_car", "calib_arith_sigma_bias", "filter_car_sigma_bias", "acc_bits")}
for sp in specs:
    w = load_window(window_path(sp["pid"], sp["t0_s"], sp["duration_s"])); codes = adc_quantize(w["voltages_uv"], fs); vq = codes * 2.34375
    ff = filter_causal(vq, V2); fx = run_fixed(codes, fs); fl = run_detector(vq, V2, w["coordinates_um"])
    fi = fx["filtered"] * ilsb; fq = signal.sosfilt(fx["sos"].astype(float) / 2 ** 30, vq, axis=0)
    agg["filter_total_rms_uv"].append(rms(fi - ff)); agg["filter_coef_effect_rms_uv"].append(rms(fq - ff)); agg["filter_trunc_effect_rms_uv"].append(rms(fi - fq))
    agg["filter_max_abs_uv"].append(float(np.abs(fi - ff).max())); agg["gain_int_vs_float_q"].append(gain(fi, fq))
    agg["gain_car"].append(gain(fx["car"] * ilsb, common_reference(ff, "median")))
    car_x = fx["car"] * ilsb
    _, s_fx = estimate_noise_variant(car_x, V2); s_int = fx["calibration"]["sigma_q16"] / 65536 * ilsb
    agg["calib_arith_sigma_bias"].append(float(((s_int - s_fx) / s_fx).mean())); agg["filter_car_sigma_bias"].append(float(((s_fx - fl["sigma"]) / fl["sigma"]).mean()))
    agg["acc_bits"].append(max(s["accumulator_bits_required_including_sign"] for s in fx["filter_stats"]))
out = {"windows": [f"{s['session']}:{s['t0_s']}" for s in specs], "means": {k: float(np.mean(v)) for k, v in agg.items()}, "max": {k: float(np.max(v)) for k, v in agg.items()}}
# outlier window with large shared artifact (A, selection)
sp = [s for s in manifest_windows("selection") if s["session"] == "A" and abs(s["t0_s"] - 1989.596) < 1e-3]
if sp:
    sp = sp[0]; w = load_window(window_path(sp["pid"], sp["t0_s"], sp["duration_s"])); codes = adc_quantize(w["voltages_uv"], fs)
    fx = run_fixed(codes, fs); e = fx["filtered"] * ilsb - filter_causal(codes * 2.34375, V2)
    out["artifact_window_A_1989s"] = {"code_std_median_channel": float(np.median(codes.std(0))), "filter_error_rms_uv": rms(e), "filter_error_max_uv": float(np.abs(e).max()), "acc_bits": max(s["accumulator_bits_required_including_sign"] for s in fx["filter_stats"])}
# analytic bound
def bound(hp, lp):
    from scripts.filter_iir import IirConfig, design_sos
    from scripts.fixed_point_model import FixedPointConfig, quantize_sos
    q = quantize_sos(design_sos(IirConfig(sample_rate_hz=30000., highpass_hz=hp, lowpass_hz=lp)), FixedPointConfig(signal_bits=24)).astype(float) / 2 ** 30
    imp = np.zeros(60000); imp[0] = 1
    ia1 = signal.lfilter([1.0], q[0, 3:6], imp); ia2 = signal.lfilter([1.0], q[1, 3:6], imp); h1 = signal.lfilter(q[1, 0:3], q[1, 3:6], ia1)
    return float((np.abs(h1).sum() + np.abs(ia2).sum()) * 2.34375 / 1024)
out["analytic_worst_case_truncation_error_uv"] = {"v2_200_4500": bound(200., 4500.), "baseline_300_6000": bound(300., 6000.), "at_12_guard_bits_v2": bound(200., 4500.) / 4}
# rounding-rule diagnostic on 4 channels of one window
w = load_window(window_path(specs[0]["pid"], specs[0]["t0_s"], specs[0]["duration_s"])); codes = adc_quantize(w["voltages_uv"], fs)[:, :4]
sos = [tuple(int(v) for v in r) for r in run_fixed(codes, fs)["sos"]]; ref = signal.sosfilt(np.array(sos, float) / 2 ** 30, codes * 2.34375, axis=0)
def run(rule, col, G=10):
    st = [[0, 0, 0, 0] for _ in sos]; o = []
    for c in col:
        x = int(c) << G
        for k, (b0, b1, b2, _, a1, a2) in enumerate(sos):
            x1, x2, y1, y2 = st[k]; acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y = (acc + (1 << 29)) >> 30 if rule == "round_nearest" else (acc >> 30 if rule == "floor" or acc >= 0 else -((-acc) >> 30))
            st[k] = [x, x1, y, y1]; x = y
        o.append(x)
    return np.array(o)
rr = {}
for rule in ("trunc_toward_zero", "round_nearest", "floor"):
    ys = [run(rule, codes[:, c]) * ilsb for c in range(4)]
    rr[rule] = {"gain_error": float(np.mean([gain(ys[c], ref[:, c]) for c in range(4)])), "rms_error_uv": float(np.mean([rms(ys[c] - ref[:, c]) for c in range(4)]))}
burst = np.concatenate([np.random.default_rng(1).integers(-512, 512, 1500), np.zeros(20000, dtype=int)])
for rule in ("trunc_toward_zero", "round_nearest"):
    rr[rule]["zero_input_residual_after_burst_internal_lsb"] = int(np.abs(run(rule, burst)[-2000:]).max())
out["rounding_rule_diagnostic"] = rr
(ROOT / "reports" / "phase6" / "error_decomposition.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
