"""Generate dense, seeded ADC streams and exact Python pipeline events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.detect_spikes import DetectorConfig
from scripts.filter_iir import IirConfig, design_sos
from scripts.fixed_point_model import FixedPointConfig, fixed_iir_filter, quantize_sos
from scripts.fixed_point_model import detect_integer_candidates, merge_integer_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--prefix", type=Path, default=Path("/tmp/neurosig-stress"))
    args = parser.parse_args()
    if not 1 <= args.channels <= 32 or args.frames < 80:
        parser.error("channels must be 1..32 and frames at least 80")
    rng = np.random.default_rng(args.seed)
    codes = rng.integers(-250, 251, size=(args.frames, args.channels), dtype=np.int16)
    for channel in range(args.channels):
        for frame in range(24 + channel % 7, args.frames - 24, 45):
            codes[frame, channel] = -18000 - 100 * channel
    coeff = quantize_sos(design_sos(IirConfig()), FixedPointConfig())
    filtered, _ = fixed_iir_filter(codes, coeff, FixedPointConfig())
    thresholds = np.full(args.channels, -3000, dtype=np.int32)
    candidates = detect_integer_candidates(filtered, thresholds, DetectorConfig())
    events = merge_integer_events(candidates, DetectorConfig())
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{args.prefix}-adc.txt").open("w") as out:
        for code in codes.flat:
            out.write(f"{int(code) & 0xffff:04x}\n")
    with Path(f"{args.prefix}-thresholds.txt").open("w") as out:
        out.writelines(f"{int(t) & 0xffffff:06x}\n" for t in thresholds)
    with Path(f"{args.prefix}-events.txt").open("w") as out:
        out.write(f"{len(events)}\n")
        for event in events:
            out.write(f"{event.sample} {event.channel} {event.code}\n")
    print(f"seed={args.seed} frames={args.frames} channels={args.channels} "
          f"candidates={len(candidates)} merged={len(events)}")


if __name__ == "__main__":
    main()
