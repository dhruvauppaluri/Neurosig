"""Export calibrated thresholds and Python merged events for RTL pipeline test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.detect_spikes import DetectorConfig
from scripts.fixed_point_model import detect_integer_candidates, merge_integer_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=Path("data/fixed_point/filtered_codes_i24.npy"))
    parser.add_argument("--thresholds", type=Path, default=Path("data/fixed_point/threshold_codes.npy"))
    parser.add_argument("--threshold-output", type=Path, default=Path("/tmp/neurosig-thresholds.txt"))
    parser.add_argument("--events-output", type=Path, default=Path("/tmp/neurosig-merged-events.txt"))
    args = parser.parse_args()
    filtered = np.load(args.filtered)
    thresholds = np.load(args.thresholds)
    if filtered.shape != (32000, 32) or thresholds.shape != (32,):
        raise ValueError("expected the 32,000-frame, 32-channel recording")
    candidates = detect_integer_candidates(filtered, thresholds, DetectorConfig())
    events = merge_integer_events(candidates, DetectorConfig())
    args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    with args.threshold_output.open("w") as handle:
        for code in thresholds:
            handle.write(f"{int(code) & 0xffffff:06x}\n")
    with args.events_output.open("w") as handle:
        handle.write(f"{len(events)}\n")
        for event in events:
            handle.write(f"{event.sample} {event.channel} {event.code}\n")
    print(f"Wrote 32 thresholds and {len(events)} merged events from {len(candidates)} candidates")


if __name__ == "__main__":
    main()
