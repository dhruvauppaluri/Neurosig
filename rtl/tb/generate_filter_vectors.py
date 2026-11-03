"""Export sample-major MEArec ADC inputs and Python fixed-point filter outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.filter_iir import IirConfig, design_sos
from scripts.fixed_point_model import FixedPointConfig, fixed_iir_filter, quantize_sos

RTL_COEFFICIENTS = np.array([
    [1029936539, -2059873079, 1029936539, 1 << 30, -2058085217, 987919117],
    [200461514, 400923028, 200461514, 1 << 30, -497075920, 225180151],
], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, default=Path("data/quantized/adc_codes.npy"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/neurosig-filter-vectors.txt"))
    parser.add_argument("--frames", type=int, default=1024)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    config = FixedPointConfig()
    coefficients = quantize_sos(design_sos(IirConfig()), config)
    if not np.array_equal(coefficients, RTL_COEFFICIENTS):
        raise RuntimeError("RTL coefficient constants differ from the Python reference")
    codes = np.load(args.codes, mmap_mode="r")
    if codes.ndim != 2 or codes.shape[1] != 32 or len(codes) < args.frames:
        raise ValueError("expected at least --frames rows of 32-channel ADC codes")
    inputs = np.asarray(codes[:args.frames], dtype=np.int64)
    expected, _ = fixed_iir_filter(inputs, coefficients, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for frame in range(args.frames):
            for channel in range(32):
                handle.write(
                    f"{channel} {int(inputs[frame, channel]) & 0xffff:04x} "
                    f"{int(expected[frame, channel]) & 0xffffff:06x}\n"
                )
    print(f"Wrote {inputs.size} vectors ({args.frames} frames × 32 channels) to {args.output}")


if __name__ == "__main__":
    main()
