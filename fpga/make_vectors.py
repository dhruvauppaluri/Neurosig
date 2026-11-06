"""Generate small deterministic 32-channel ADC playback fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("zeros", "impulse", "pattern"), required=True)
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.frames < 64:
        p.error("at least 64 frames are required for a useful flush test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for frame in range(args.frames):
            for channel in range(32):
                if args.kind == "zeros":
                    code = 0
                elif args.kind == "impulse":
                    code = -12000 if frame == 32 and channel == 21 else 0
                else:
                    code = ((frame * 37 + channel * 19) % 401) - 200
                output.write(struct.pack("<h", code))
    print(f"Wrote {args.frames * 32} ADC samples to {args.output}")


if __name__ == "__main__":
    main()
