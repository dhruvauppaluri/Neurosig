"""Quantize a MEArec voltage recording into signed ADC codes.

The default configuration implements the current Neurosig design decision:
16-bit two's-complement codes covering -135 uV through one LSB below +135 uV.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class AdcConfig:
    bits: int = 16
    full_scale_uv: float = 135.0

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise ValueError("bits must be between 2 and 16")
        if self.full_scale_uv <= 0:
            raise ValueError("full_scale_uv must be positive")

    @property
    def code_min(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def code_max(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def lsb_uv(self) -> float:
        return (2.0 * self.full_scale_uv) / (1 << self.bits)

    @property
    def maximum_representable_uv(self) -> float:
        return self.code_max * self.lsb_uv


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Round halves away from zero, a rule that is straightforward to match in RTL."""
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def quantize_voltages(
    voltages_uv: np.ndarray, config: AdcConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed integer codes and a mask showing which samples saturated."""
    voltages = np.asarray(voltages_uv, dtype=np.float64)
    if not np.isfinite(voltages).all():
        raise ValueError("input contains NaN or infinite voltage samples")

    rounded_codes = round_half_away_from_zero(voltages / config.lsb_uv)
    saturated = (rounded_codes < config.code_min) | (rounded_codes > config.code_max)
    codes = np.clip(rounded_codes, config.code_min, config.code_max).astype(np.int16)
    return codes, saturated


def write_hex_vectors(path: Path, codes: np.ndarray, bits: int) -> None:
    """Write one two's-complement code per line in sample-major/channel-minor order."""
    mask = (1 << bits) - 1
    width = (bits + 3) // 4
    flattened = codes.reshape(-1)
    with path.open("w") as output:
        for start in range(0, flattened.size, 65_536):
            block = flattened[start : start + 65_536]
            output.writelines(f"{int(code) & mask:0{width}X}\n" for code in block)


def build_summary(
    input_path: Path,
    voltages: np.ndarray,
    codes: np.ndarray,
    saturated: np.ndarray,
    config: AdcConfig,
) -> dict:
    reconstructed = codes.astype(np.float64) * config.lsb_uv
    error = reconstructed - voltages.astype(np.float64)
    unclipped_error = error[~saturated]
    return {
        "input": str(input_path),
        "shape": list(voltages.shape),
        "adc": {
            **asdict(config),
            "signed": True,
            "encoding": "two's complement",
            "code_min": config.code_min,
            "code_max": config.code_max,
            "lsb_uv": config.lsb_uv,
            "minimum_representable_uv": config.code_min * config.lsb_uv,
            "maximum_representable_uv": config.maximum_representable_uv,
            "rounding": "nearest, halves away from zero",
            "overflow": "saturate",
        },
        "stream_order": "sample-major, then channel-minor",
        "input_min_uv": float(np.min(voltages)),
        "input_max_uv": float(np.max(voltages)),
        "output_code_min": int(np.min(codes)),
        "output_code_max": int(np.max(codes)),
        "saturated_samples": int(np.count_nonzero(saturated)),
        "saturated_fraction": float(np.mean(saturated)),
        "quantization_error_uv": {
            "maximum_absolute_unclipped": float(np.max(np.abs(unclipped_error))),
            "mean_absolute_unclipped": float(np.mean(np.abs(unclipped_error))),
            "rms_unclipped": float(np.sqrt(np.mean(unclipped_error**2))),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/unfiltered/recording.h5"),
        help="MEArec HDF5 recording (default: data/unfiltered/recording.h5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/quantized"),
        help="output directory (default: data/quantized)",
    )
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--full-scale-uv", type=float, default=135.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AdcConfig(bits=args.bits, full_scale_uv=args.full_scale_uv)

    with h5py.File(args.input, "r") as recording_file:
        if "recordings" not in recording_file:
            raise KeyError(f"{args.input} has no 'recordings' dataset")
        voltages = recording_file["recordings"][()]
    if voltages.ndim != 2:
        raise ValueError(f"expected [samples, channels], got shape {voltages.shape}")

    codes, saturated = quantize_voltages(voltages, config)
    summary = build_summary(args.input, voltages, codes, saturated, config)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "adc_codes.npy", codes)
    codes.astype("<i2", copy=False).tofile(args.output / "adc_codes_i16le.bin")
    write_hex_vectors(args.output / "adc_codes.hex", codes, config.bits)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote quantized outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
