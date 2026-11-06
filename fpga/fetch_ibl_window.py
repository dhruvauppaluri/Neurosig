"""Save a small public IBL AP window and channel geometry for C5G playback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.ibl import DEFAULT_PID, load_raw_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", default=DEFAULT_PID)
    parser.add_argument("--start-s", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start_s < 0 or args.duration_s <= 0:
        parser.error("start time must be nonnegative and duration positive")
    chunk = load_raw_chunk(args.pid, args.start_s, args.duration_s, 32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    voltage_path = args.output_dir / "window_uv.npy"
    position_path = args.output_dir / "channel_positions_um.npy"
    np.save(voltage_path, chunk["voltages_uv"])
    positions = np.asarray(chunk["locations"])[chunk["selected"]]
    np.save(position_path, np.column_stack((positions, np.zeros(len(positions)))))
    metadata = {
        "pid": args.pid,
        "eid": str(chunk["eid"]),
        "probe": chunk["probe"],
        "start_s": args.start_s,
        "duration_s": args.duration_s,
        "sample_rate_hz": chunk["fs"],
        "selected_sorted_channel_indices": chunk["selected"].tolist(),
        "selected_raw_channel_indices": chunk["raw_indices"].tolist(),
        "shape": list(chunk["voltages_uv"].shape),
        "unit": "microvolts",
        "source": "public IBL calibrated AP stream; no CAR applied",
    }
    (args.output_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
