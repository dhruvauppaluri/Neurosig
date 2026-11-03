"""Export a one-channel filtered trace and Python detector events for RTL comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.detect_spikes import DetectorConfig
from scripts.fixed_point_model import detect_integer_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=Path("data/fixed_point/filtered_codes_i24.npy"))
    parser.add_argument("--thresholds", type=Path, default=Path("data/fixed_point/threshold_codes.npy"))
    parser.add_argument("--channel", type=int, default=21)
    parser.add_argument("--samples", type=int, default=32000)
    parser.add_argument("--trace-output", type=Path, default=Path("/tmp/neurosig-detector-trace.txt"))
    parser.add_argument("--events-output", type=Path, default=Path("/tmp/neurosig-detector-events.txt"))
    args = parser.parse_args()
    filtered = np.load(args.filtered, mmap_mode="r")
    thresholds = np.load(args.thresholds)
    if filtered.ndim != 2 or thresholds.shape != (filtered.shape[1],):
        raise ValueError("filtered trace and threshold shape mismatch")
    if not 0 <= args.channel < filtered.shape[1]:
        parser.error("channel outside recording")
    if not 17 <= args.samples <= filtered.shape[0]:
        parser.error("samples must be 17 through recording length")
    values = np.asarray(filtered[:args.samples, args.channel], dtype=np.int32)
    threshold = int(thresholds[args.channel])
    # The reference accepts [samples, channels]. This narrow view preserves the
    # original sample indices while selecting a single channel.
    expected = detect_integer_candidates(values[:, None], np.array([threshold]), DetectorConfig())
    if expected and expected[-1].sample + 16 >= len(values):
        raise ValueError("last candidate search would be truncated at end of trace")
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_output.open("w") as handle:
        handle.write(f"{threshold} {len(values)}\n")
        for code in values:
            handle.write(f"{int(code) & 0xffffff:06x}\n")
    with args.events_output.open("w") as handle:
        handle.write(f"{len(expected)}\n")
        for event in expected:
            handle.write(f"{event.sample} {event.code}\n")
    print(f"Channel {args.channel}: {len(values)} samples, threshold {threshold}, "
          f"{len(expected)} Python candidate events")


if __name__ == "__main__":
    main()
