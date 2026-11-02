"""Run reproducible randomized stress tests on the fixed-point reference model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import signal

from scripts.detect_spikes import DetectorConfig
from scripts.filter_iir import IirConfig, design_sos, maximum_pole_magnitude
from scripts.fixed_point_model import (
    FixedPointConfig,
    dequantize_sos,
    detect_integer_candidates,
    fixed_iir_filter,
    merge_integer_events,
    quantize_sos,
)


DEFAULT_SEED = 20_261_025
DEFAULT_TRIALS = 100
ADC_LSB_UV = 270.0 / 65_536.0
RMS_ERROR_LIMIT_CODES = 64.0
MAX_ERROR_LIMIT_CODES = 256.0
ZERO_TAIL_SAMPLES = 2_048
ZERO_SETTLE_WINDOW = 256


def generate_filter_input(
    rng: np.random.Generator, samples: int, channels: int, pattern: int
) -> np.ndarray:
    """Generate bounded int16 inputs with different difficult signal shapes."""
    if pattern == 0:  # Gaussian noise from quiet to nearly full scale.
        standard_deviation = rng.uniform(10.0, 20_000.0)
        values = rng.normal(0.0, standard_deviation, size=(samples, channels))
    elif pattern == 1:  # Several in-band and out-of-band tones.
        time = np.arange(samples, dtype=np.float64) / 32_000.0
        values = np.zeros((samples, channels), dtype=np.float64)
        frequencies = np.array([50.0, 300.0, 1_000.0, 6_000.0, 12_000.0])
        for channel in range(channels):
            amplitudes = rng.uniform(500.0, 5_000.0, size=len(frequencies))
            phases = rng.uniform(0.0, 2.0 * np.pi, size=len(frequencies))
            values[:, channel] = np.sum(
                amplitudes[:, None]
                * np.sin(2.0 * np.pi * frequencies[:, None] * time + phases[:, None]),
                axis=0,
            )
    elif pattern == 2:  # Full-scale impulses with both polarities.
        values = np.zeros((samples, channels), dtype=np.float64)
        for channel in range(channels):
            count = int(rng.integers(1, 9))
            locations = rng.choice(samples, size=count, replace=False)
            values[locations, channel] = rng.choice([-32_768, 32_767], size=count)
    elif pattern == 3:  # Abrupt steps exercise filter transients and feedback.
        values = np.zeros((samples, channels), dtype=np.float64)
        for channel in range(channels):
            boundaries = np.sort(rng.choice(np.arange(1, samples), size=5, replace=False))
            levels = rng.integers(-32_768, 32_768, size=6)
            for level, (start, stop) in zip(
                levels, zip(np.r_[0, boundaries], np.r_[boundaries, samples])
            ):
                values[start:stop, channel] = level
    elif pattern == 4:  # Repeated extreme and near-extreme ADC codes.
        alphabet = np.array([-32_768, -32_767, -1, 0, 1, 32_766, 32_767])
        values = rng.choice(alphabet, size=(samples, channels))
    else:  # Noise plus short negative spike-like pulses.
        values = rng.normal(0.0, rng.uniform(200.0, 2_000.0), size=(samples, channels))
        for _ in range(int(rng.integers(1, 10))):
            center = int(rng.integers(3, samples - 3))
            channel = int(rng.integers(channels))
            amplitude = float(rng.uniform(5_000.0, 30_000.0))
            values[center - 2 : center + 3, channel] -= amplitude * np.array(
                [0.15, 0.55, 1.0, 0.55, 0.15]
            )
    return np.clip(np.rint(values), -32_768, 32_767).astype(np.int16)


def exercise_detector(
    rng: np.random.Generator, samples: int, channels: int, config: DetectorConfig
) -> tuple[int, int]:
    """Generate known integer events and require exact detector/merge results."""
    thresholds = -rng.integers(200, 1_001, size=channels, dtype=np.int32)
    traces = rng.integers(-5, 6, size=(samples, channels), dtype=np.int32)
    merge_samples = config.samples_for_ms(config.merge_ms)
    refractory_samples = config.samples_for_ms(config.refractory_ms)
    spacing = max(4 * refractory_samples, 4 * merge_samples, 64)
    centers = np.arange(spacing, samples - spacing, spacing)
    if len(centers):
        selected = centers[rng.random(len(centers)) < 0.65]
    else:
        selected = np.empty(0, dtype=int)

    expected: list[tuple[int, int, int, int]] = []
    for center in selected:
        visible_count = int(rng.integers(1, channels + 1))
        visible_channels = rng.choice(channels, size=visible_count, replace=False)
        offsets = rng.integers(0, merge_samples + 1, size=visible_count)
        leader_index = int(rng.integers(visible_count))
        margins = rng.integers(250, 1_000, size=visible_count)
        margins[leader_index] = 2_000
        for index, channel in enumerate(visible_channels):
            peak = int(center + offsets[index])
            threshold = int(thresholds[channel])
            code = threshold - int(margins[index])
            traces[peak - 1, channel] = threshold + 20
            traces[peak, channel] = code
            traces[peak + 1, channel] = threshold + 20
        leader_channel = int(visible_channels[leader_index])
        leader_sample = int(center + offsets[leader_index])
        leader_code = int(thresholds[leader_channel]) - 2_000
        expected.append((leader_sample, leader_channel, leader_code, 2_000))

    candidates = detect_integer_candidates(traces, thresholds, config)
    merged = merge_integer_events(candidates, config)
    actual = [
        (event.sample, event.channel, event.code, event.threshold_margin_codes)
        for event in merged
    ]
    if actual != expected:
        raise AssertionError(f"detector mismatch: expected {expected}, got {actual}")
    return len(candidates), len(merged)


def run_stress(seed: int, trials: int) -> dict:
    rng = np.random.default_rng(seed)
    fixed_config = FixedPointConfig()
    detector_config = DetectorConfig()
    float_sos = design_sos(IirConfig())
    quantized_sos = quantize_sos(float_sos, fixed_config)
    dequantized_sos = dequantize_sos(quantized_sos, fixed_config)
    maximum_pole = maximum_pole_magnitude(dequantized_sos)
    if maximum_pole >= 1.0:
        raise AssertionError(f"quantized filter is unstable: pole magnitude {maximum_pole}")

    trial_reports: list[dict] = []
    for trial in range(trials):
        samples = int(rng.integers(768, 2_049))
        channels = int(rng.integers(1, 9))
        pattern = trial % 6
        excitation = generate_filter_input(rng, samples, channels, pattern)
        input_codes = np.vstack(
            (excitation, np.zeros((ZERO_TAIL_SAMPLES, channels), dtype=np.int16))
        )

        fixed_output, section_stats = fixed_iir_filter(
            input_codes, quantized_sos, fixed_config
        )
        float_output = signal.sosfilt(dequantized_sos, input_codes.astype(np.float64), axis=0)
        error = fixed_output.astype(np.float64) - float_output
        rms_error = float(np.sqrt(np.mean(np.square(error))))
        maximum_error = float(np.max(np.abs(error)))
        saturation_count = sum(stat["saturated_samples"] for stat in section_stats)
        accumulator_bits = max(
            stat["accumulator_bits_required_including_sign"] for stat in section_stats
        )

        if saturation_count != 0:
            raise AssertionError(f"trial {trial}: unexpected signal saturation")
        if accumulator_bits > fixed_config.accumulator_bits:
            raise AssertionError(f"trial {trial}: accumulator width exceeded")
        if not np.all(fixed_output[-ZERO_SETTLE_WINDOW:] == 0):
            raise AssertionError(f"trial {trial}: zero-input response did not settle")
        if rms_error > RMS_ERROR_LIMIT_CODES:
            raise AssertionError(
                f"trial {trial}: RMS error {rms_error:.3f} exceeds {RMS_ERROR_LIMIT_CODES} codes"
            )
        if maximum_error > MAX_ERROR_LIMIT_CODES:
            raise AssertionError(
                f"trial {trial}: max error {maximum_error:.3f} exceeds {MAX_ERROR_LIMIT_CODES} codes"
            )
        if trial % 10 == 0:
            repeated, _ = fixed_iir_filter(input_codes, quantized_sos, fixed_config)
            if not np.array_equal(fixed_output, repeated):
                raise AssertionError(f"trial {trial}: fixed model is not deterministic")

        detector_candidates, detector_events = exercise_detector(
            rng, samples, channels, detector_config
        )
        trial_reports.append(
            {
                "trial": trial,
                "pattern": pattern,
                "samples": samples,
                "channels": channels,
                "rms_error_codes": rms_error,
                "maximum_error_codes": maximum_error,
                "saturation_count": saturation_count,
                "accumulator_bits_required": accumulator_bits,
                "detector_candidates": detector_candidates,
                "merged_detector_events": detector_events,
            }
        )

    worst_rms = max(trial_reports, key=lambda report: report["rms_error_codes"])
    worst_max = max(trial_reports, key=lambda report: report["maximum_error_codes"])
    return {
        "result": "pass",
        "seed": seed,
        "trials_requested": trials,
        "trials_passed": len(trial_reports),
        "coverage": {
            "input_patterns": [
                "Gaussian noise",
                "multitone",
                "full-scale impulses",
                "abrupt steps",
                "extreme-code sequence",
                "noise plus negative pulses",
            ],
            "channels": "random 1 through 8 per trial",
            "samples": "random 768 through 2048, plus 2048 zero samples",
            "checks": [
                "quantized poles remain inside the unit circle",
                "no 24-bit section saturation",
                "64-bit accumulator is not exceeded",
                "last 256 zero-input outputs are exactly zero",
                "fixed-versus-float error bounds",
                "repeat-run determinism every tenth trial",
                "exact randomized detector crossings, refractory, and merge behavior",
            ],
        },
        "limits": {
            "rms_error_codes": RMS_ERROR_LIMIT_CODES,
            "maximum_error_codes": MAX_ERROR_LIMIT_CODES,
            "rms_error_uv": RMS_ERROR_LIMIT_CODES * ADC_LSB_UV,
            "maximum_error_uv": MAX_ERROR_LIMIT_CODES * ADC_LSB_UV,
        },
        "measured": {
            "maximum_quantized_pole_magnitude": maximum_pole,
            "worst_rms_error_codes": worst_rms["rms_error_codes"],
            "worst_rms_error_uv": worst_rms["rms_error_codes"] * ADC_LSB_UV,
            "worst_rms_trial": worst_rms["trial"],
            "worst_absolute_error_codes": worst_max["maximum_error_codes"],
            "worst_absolute_error_uv": worst_max["maximum_error_codes"] * ADC_LSB_UV,
            "worst_absolute_error_trial": worst_max["trial"],
            "maximum_accumulator_bits_required": max(
                report["accumulator_bits_required"] for report in trial_reports
            ),
            "total_detector_candidates": sum(
                report["detector_candidates"] for report in trial_reports
            ),
            "total_merged_detector_events": sum(
                report["merged_detector_events"] for report in trial_reports
            ),
            "total_saturations": sum(report["saturation_count"] for report in trial_reports),
        },
        "configuration": {
            "fixed_point": asdict(fixed_config),
            "detector": asdict(detector_config),
        },
        "trials": trial_reports,
        "scope_note": (
            "Randomized simulation increases confidence in this software arithmetic model; "
            "it does not replace RTL equivalence testing, formal verification, or broader neural datasets."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/fixed-point-stress.json")
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = run_stress(args.seed, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["result", "seed", "trials_passed", "measured"]}, indent=2))
    print(f"Wrote randomized stress report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
