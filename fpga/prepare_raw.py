"""Prepare a 32-channel raw-voltage window for no-CAR C5G playback.

Input is a .npy matrix in microvolts. A 30 kHz recording can be resampled to
the RTL's fixed 32 kHz configuration. The output reference uses the existing
bit-accurate Python filter and integer detector with offline 6-sigma thresholds.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.detect_spikes import DetectorConfig
from scripts.filter_iir import IirConfig, design_sos
from scripts.fixed_point_model import (FixedPointConfig, build_threshold_codes,
                                       detect_integer_candidates, fixed_iir_filter,
                                       merge_integer_events, quantize_sos)
from scripts.quantize_adc import AdcConfig, quantize_voltages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(input_path: Path, output_dir: Path, *, sample_rate_hz: float,
            full_scale_uv: float, channel_indices: list[int] | None,
            allow_saturation: bool = False) -> dict:
    raw = np.load(input_path, mmap_mode="r")
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("input must be a numeric [samples, channels] .npy array")
    if channel_indices is None:
        if raw.shape[1] != 32:
            raise ValueError("select exactly 32 channels from a wider raw recording")
        channel_indices = list(range(32))
    if len(channel_indices) != 32 or len(set(channel_indices)) != 32 or min(channel_indices) < 0 or max(channel_indices) >= raw.shape[1]:
        raise ValueError("channel selection must contain 32 distinct valid indices")
    samples_uv = np.asarray(raw[:, channel_indices], dtype=np.float64)
    if not np.isfinite(samples_uv).all():
        raise ValueError("raw recording contains non-finite voltages")
    if sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    if sample_rate_hz != 32_000:
        ratio = Fraction(32_000 / sample_rate_hz).limit_denominator(1000)
        if abs(float(ratio) - 32_000 / sample_rate_hz) > 1e-9:
            raise ValueError("sample-rate ratio needs a higher precision resampler")
        samples_uv = signal.resample_poly(samples_uv, ratio.numerator,
                                          ratio.denominator, axis=0)
    else:
        ratio = Fraction(1, 1)
    adc = AdcConfig(full_scale_uv=full_scale_uv)
    codes, saturated = quantize_voltages(samples_uv, adc)
    saturation_count = int(np.count_nonzero(saturated))
    if saturation_count and not allow_saturation:
        raise ValueError(f"{saturation_count} ADC values saturate; adjust --full-scale-uv")
    sos = design_sos(IirConfig())
    fixed_cfg = FixedPointConfig()
    quantized_sos = quantize_sos(sos, fixed_cfg)
    # Calibrate from the reconstructed ADC stream, matching filter_iir.py.
    # Calibration on pre-quantized voltages can shift thresholds materially.
    float_filtered = signal.sosfilt(sos, codes.astype(np.float64) * adc.lsb_uv,
                                    axis=0)  # no CAR
    detector = DetectorConfig()
    thresholds, _baseline, _noise = build_threshold_codes(
        float_filtered, adc.lsb_uv, detector, fixed_cfg)
    fixed_filtered, section_stats = fixed_iir_filter(codes, quantized_sos, fixed_cfg)
    events = merge_integer_events(
        detect_integer_candidates(fixed_filtered, thresholds, detector), detector)
    output_dir.mkdir(parents=True, exist_ok=True)
    adc_path = output_dir / "adc_i16le.bin"
    threshold_path = output_dir / "thresholds.hex"
    expected_path = output_dir / "expected.json"
    codes.astype("<i2", copy=False).tofile(adc_path)
    threshold_path.write_text("".join(f"{int(code) & 0xffffff:06X}\n" for code in thresholds))
    expected = {
        "reference": "fixed_point_no_car_6sigma_300_6000_negative",
        "input": str(input_path), "input_sha256": sha256(input_path),
        "input_sample_rate_hz": sample_rate_hz,
        "output_sample_rate_hz": 32_000,
        "resample_up": ratio.numerator, "resample_down": ratio.denominator,
        "channel_indices": channel_indices,
        "full_scale_uv": full_scale_uv, "lsb_uv": adc.lsb_uv,
        "adc_saturated_values": saturation_count,
        "samples": int(codes.size), "frames": int(codes.shape[0]),
        "section_stats": section_stats,
        "events": [{"sample": e.sample, "channel": e.channel, "code": e.code,
                    "threshold_margin_codes": e.threshold_margin_codes} for e in events],
        "adc_sha256": sha256(adc_path), "thresholds_sha256": sha256(threshold_path),
    }
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    return {"adc": str(adc_path), "thresholds": str(threshold_path),
            "expected": str(expected_path), "frames": int(codes.shape[0]),
            "events": len(events), "saturated_values": saturation_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npy", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--full-scale-uv", type=float, required=True)
    parser.add_argument("--channels-file", type=Path,
                        help="JSON array of 32 source channel indices")
    parser.add_argument("--allow-saturation", action="store_true")
    args = parser.parse_args()
    indices = json.loads(args.channels_file.read_text()) if args.channels_file else None
    print(json.dumps(prepare(args.input_npy, args.output_dir,
                             sample_rate_hz=args.sample_rate_hz,
                             full_scale_uv=args.full_scale_uv,
                             channel_indices=indices,
                             allow_saturation=args.allow_saturation), indent=2))


if __name__ == "__main__":
    main()
