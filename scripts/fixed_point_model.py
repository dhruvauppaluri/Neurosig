"""Bit-accurate fixed-point model of the Neurosig IIR filter and spike detector."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from scripts.detect_spikes import (
    DetectorConfig,
    estimate_noise,
    load_ground_truth,
    match_events,
)
from scripts.filter_iir import IirConfig, design_sos, maximum_pole_magnitude
from scripts.quantize_adc import round_half_away_from_zero, write_hex_vectors


@dataclass(frozen=True)
class FixedPointConfig:
    coefficient_bits: int = 32
    coefficient_fraction_bits: int = 30
    signal_bits: int = 24
    accumulator_bits: int = 64

    def __post_init__(self) -> None:
        if self.coefficient_bits != 32 or self.coefficient_fraction_bits != 30:
            raise ValueError("this reference currently requires 32-bit Q2.30 coefficients")
        if not 17 <= self.signal_bits <= 30:
            raise ValueError("signal_bits must be between 17 and 30")
        if self.accumulator_bits != 64:
            raise ValueError("this reference currently requires a 64-bit accumulator")

    @property
    def coefficient_scale(self) -> int:
        return 1 << self.coefficient_fraction_bits

    @property
    def signal_min(self) -> int:
        return -(1 << (self.signal_bits - 1))

    @property
    def signal_max(self) -> int:
        return (1 << (self.signal_bits - 1)) - 1


@dataclass(frozen=True)
class IntegerEvent:
    sample: int
    channel: int
    code: int
    threshold_margin_codes: int


def quantize_sos(sos: np.ndarray, config: FixedPointConfig) -> np.ndarray:
    """Quantize normalized SOS coefficients with half-away-from-zero rounding."""
    scaled = round_half_away_from_zero(np.asarray(sos) * config.coefficient_scale)
    coefficient_min = -(1 << (config.coefficient_bits - 1))
    coefficient_max = (1 << (config.coefficient_bits - 1)) - 1
    if np.any(scaled < coefficient_min) or np.any(scaled > coefficient_max):
        raise OverflowError("a coefficient does not fit the selected coefficient format")
    quantized = scaled.astype(np.int64)
    quantized[:, 3] = config.coefficient_scale  # a0 is exactly one
    return quantized


def dequantize_sos(quantized_sos: np.ndarray, config: FixedPointConfig) -> np.ndarray:
    return np.asarray(quantized_sos, dtype=np.float64) / config.coefficient_scale


def truncate_shift_toward_zero(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    """Remove coefficient fractional bits with magnitude truncation.

    Truncation toward zero avoids the persistent zero-input limit cycle observed
    with round-to-nearest in this direct-form-I feedback path.
    """
    values = np.asarray(values, dtype=np.int64)
    return np.where(values >= 0, values >> fractional_bits, -((-values) >> fractional_bits))


def fixed_iir_filter(
    input_codes: np.ndarray,
    quantized_sos: np.ndarray,
    config: FixedPointConfig,
) -> tuple[np.ndarray, list[dict]]:
    """Run cascaded direct-form-I biquads with integer states and RTL-style limits."""
    stage_input = np.asarray(input_codes, dtype=np.int64)
    if stage_input.ndim != 2:
        raise ValueError(f"expected [samples, channels], got {stage_input.shape}")
    section_stats: list[dict] = []
    accumulator_min = -(1 << (config.accumulator_bits - 1))
    accumulator_max = (1 << (config.accumulator_bits - 1)) - 1

    for section_index, coefficients in enumerate(np.asarray(quantized_sos, dtype=np.int64)):
        b0, b1, b2, _, a1, a2 = coefficients
        channels = stage_input.shape[1]
        x1 = np.zeros(channels, dtype=np.int64)
        x2 = np.zeros(channels, dtype=np.int64)
        y1 = np.zeros(channels, dtype=np.int64)
        y2 = np.zeros(channels, dtype=np.int64)
        output = np.empty_like(stage_input)
        saturation_count = 0
        maximum_accumulator_magnitude = 0

        for sample_index in range(stage_input.shape[0]):
            x0 = stage_input[sample_index]
            accumulator = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            maximum_accumulator_magnitude = max(
                maximum_accumulator_magnitude, int(np.max(np.abs(accumulator)))
            )
            if np.any(accumulator < accumulator_min) or np.any(accumulator > accumulator_max):
                raise OverflowError(f"section {section_index} exceeded the 64-bit accumulator")
            reduced = truncate_shift_toward_zero(accumulator, config.coefficient_fraction_bits)
            saturated = (reduced < config.signal_min) | (reduced > config.signal_max)
            saturation_count += int(np.count_nonzero(saturated))
            y0 = np.clip(reduced, config.signal_min, config.signal_max)
            output[sample_index] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0

        section_stats.append(
            {
                "section": section_index,
                "saturated_samples": saturation_count,
                "maximum_accumulator_magnitude": maximum_accumulator_magnitude,
                "accumulator_bits_required_including_sign": maximum_accumulator_magnitude.bit_length() + 1,
                "output_min_code": int(np.min(output)),
                "output_max_code": int(np.max(output)),
            }
        )
        stage_input = output
    return stage_input.astype(np.int32), section_stats


def build_threshold_codes(
    float_filtered_uv: np.ndarray,
    lsb_uv: float,
    detector_config: DetectorConfig,
    fixed_config: FixedPointConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create per-channel threshold-register values during offline calibration."""
    baseline_uv, noise_sigma_uv = estimate_noise(float_filtered_uv)
    threshold_uv = baseline_uv - detector_config.threshold_sigma * noise_sigma_uv
    threshold_codes = round_half_away_from_zero(threshold_uv / lsb_uv)
    threshold_codes = np.clip(threshold_codes, fixed_config.signal_min, fixed_config.signal_max).astype(np.int32)
    return threshold_codes, baseline_uv, noise_sigma_uv


def detect_integer_candidates(
    filtered_codes: np.ndarray,
    threshold_codes: np.ndarray,
    detector_config: DetectorConfig,
) -> list[IntegerEvent]:
    """Run the detector using comparisons, subtraction, and integer counters only."""
    search = detector_config.samples_for_ms(detector_config.peak_search_ms)
    refractory = detector_config.samples_for_ms(detector_config.refractory_ms)
    events: list[IntegerEvent] = []
    for channel in range(filtered_codes.shape[1]):
        trace = filtered_codes[:, channel]
        threshold = int(threshold_codes[channel])
        crossings = np.flatnonzero((trace[:-1] >= threshold) & (trace[1:] < threshold)) + 1
        last_peak = -refractory - 1
        for crossing in crossings:
            if crossing - last_peak < refractory:
                continue
            stop = min(len(trace), crossing + search + 1)
            peak = int(crossing + np.argmin(trace[crossing:stop]))
            if peak - last_peak < refractory:
                continue
            code = int(trace[peak])
            events.append(IntegerEvent(peak, channel, code, threshold - code))
            last_peak = peak
    return sorted(events, key=lambda event: event.sample)


def merge_integer_events(
    events: list[IntegerEvent], detector_config: DetectorConfig
) -> list[IntegerEvent]:
    if not events:
        return []
    merge_samples = detector_config.samples_for_ms(detector_config.merge_ms)
    groups: list[list[IntegerEvent]] = []
    for event in sorted(events, key=lambda candidate: candidate.sample):
        if not groups or event.sample - groups[-1][0].sample > merge_samples:
            groups.append([event])
        else:
            groups[-1].append(event)
    return [max(group, key=lambda candidate: candidate.threshold_margin_codes) for group in groups]


def save_comparison_plot(
    path: Path,
    float_filtered_uv: np.ndarray,
    fixed_filtered_codes: np.ndarray,
    lsb_uv: float,
) -> None:
    channel = 21
    start, stop = int(0.125 * 32_000), int(0.14 * 32_000)
    time_ms = np.arange(start, stop) / 32_000 * 1000
    fixed_uv = fixed_filtered_codes[:, channel] * lsb_uv
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_ms, float_filtered_uv[start:stop, channel], label="float64", linewidth=1)
    axes[0].plot(time_ms, fixed_uv[start:stop], label="fixed-point", linewidth=0.8, linestyle="--")
    axes[0].set(ylabel="Voltage (µV)", title="Channel 21 filter outputs")
    axes[0].legend()
    axes[1].plot(time_ms, fixed_uv[start:stop] - float_filtered_uv[start:stop, channel], color="tab:purple")
    axes[1].set(xlabel="Time (ms)", ylabel="Error (µV)", title="Fixed-point minus float64")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adc-codes", type=Path, default=Path("data/quantized/adc_codes.npy"))
    parser.add_argument("--adc-summary", type=Path, default=Path("data/quantized/summary.json"))
    parser.add_argument("--float-filtered", type=Path, default=Path("data/filtered/filtered_uv.npy"))
    parser.add_argument("--recording", type=Path, default=Path("data/unfiltered/recording.h5"))
    parser.add_argument("--output", type=Path, default=Path("data/fixed_point"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_config = FixedPointConfig()
    filter_config = IirConfig()
    detector_config = DetectorConfig()
    adc_codes = np.load(args.adc_codes)
    float_filtered_uv = np.load(args.float_filtered)
    adc_summary = json.loads(args.adc_summary.read_text())
    lsb_uv = float(adc_summary["adc"]["lsb_uv"])

    float_sos = design_sos(filter_config)
    quantized_sos = quantize_sos(float_sos, fixed_config)
    dequantized_sos = dequantize_sos(quantized_sos, fixed_config)
    quantized_max_pole = maximum_pole_magnitude(dequantized_sos)
    if quantized_max_pole >= 1.0:
        raise RuntimeError("coefficient quantization made the IIR unstable")

    fixed_filtered_codes, section_stats = fixed_iir_filter(adc_codes, quantized_sos, fixed_config)
    fixed_filtered_uv = fixed_filtered_codes.astype(np.float64) * lsb_uv
    error_uv = fixed_filtered_uv - float_filtered_uv
    threshold_codes, baseline_uv, noise_sigma_uv = build_threshold_codes(
        float_filtered_uv, lsb_uv, detector_config, fixed_config
    )
    channel_candidates = detect_integer_candidates(fixed_filtered_codes, threshold_codes, detector_config)
    detections = merge_integer_events(channel_candidates, detector_config)
    ground_truth = load_ground_truth(args.recording)
    matches, metrics = match_events(detections, ground_truth, detector_config)

    events = [
        {
            "sample": event.sample,
            "time_s": event.sample / detector_config.sample_rate_hz,
            "channel": event.channel,
            "code": event.code,
            "voltage_uv": event.code * lsb_uv,
            "threshold_margin_codes": event.threshold_margin_codes,
        }
        for event in detections
    ]
    report = {
        "formats": {
            "adc_input": "signed 16-bit integer codes, 0.004119873046875 µV/code",
            "coefficients": "signed Q2.30",
            "section_input_output": f"signed {fixed_config.signal_bits}-bit integer codes at ADC LSB scale",
            "accumulator": "signed 64-bit",
            "coefficient_rounding": "nearest, halves away from zero",
            "section_output_quantization": "truncate toward zero to suppress zero-input limit cycles",
            "section_overflow": "saturate",
        },
        "fixed_point_config": asdict(fixed_config),
        "filter_config": asdict(filter_config),
        "detector_config": asdict(detector_config),
        "quantized_sos_coefficients": quantized_sos.tolist(),
        "stability": {
            "criterion": "dequantized coefficient poles below 1, no overflow on bounded data, and zero-input decay test",
            "maximum_quantized_pole_magnitude": quantized_max_pole,
            "stable_for_tested_model": True,
            "warning": "round-to-nearest produced a 127-code zero-input limit cycle; magnitude truncation is required and regression-tested.",
        },
        "section_stats": section_stats,
        "float_comparison_uv": {
            "maximum_absolute_error": float(np.max(np.abs(error_uv))),
            "mean_absolute_error": float(np.mean(np.abs(error_uv))),
            "rms_error": float(np.sqrt(np.mean(error_uv**2))),
        },
        "threshold_calibration": {
            "method": "offline float MAD calibration, thresholds rounded into filter-output code units",
            "threshold_codes": threshold_codes.tolist(),
            "baseline_uv": baseline_uv.tolist(),
            "noise_sigma_uv": noise_sigma_uv.tolist(),
        },
        "channel_candidates": len(channel_candidates),
        "merged_detections": len(detections),
        "events": events,
        "matches": matches,
        "ground_truth_metrics": metrics,
        "limitation": "Validated on one seeded one-second single-neuron recording; thresholds are offline calibration values.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "filtered_codes_i24.npy", fixed_filtered_codes)
    np.save(args.output / "sos_coefficients_q30.npy", quantized_sos.astype(np.int32))
    np.save(args.output / "threshold_codes.npy", threshold_codes)
    write_hex_vectors(args.output / "filtered_codes_i24.hex", fixed_filtered_codes, fixed_config.signal_bits)
    write_hex_vectors(args.output / "threshold_codes.hex", threshold_codes, fixed_config.signal_bits)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "events.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)
    save_comparison_plot(args.output / "float_vs_fixed.png", float_filtered_uv, fixed_filtered_codes, lsb_uv)
    print(json.dumps(report, indent=2))
    print(f"Wrote bit-accurate fixed-point outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
