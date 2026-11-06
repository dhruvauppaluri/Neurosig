"""Retrieve host ADC waveform windows around verified C5G event timestamps.

The FPGA transmits events only. These snippets come from the source recording,
and optional electrode positions let the existing edge pipeline localize them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.pipeline import StreamEvent, ingest, localize_events
from edge.types import Snippet


def retrieve(report: dict, adc_path: Path, *, channels: int, pre: int, post: int,
             lsb_uv: float, positions: np.ndarray | None) -> tuple[dict, dict]:
    if not report.get("comparison", {}).get("exact_match") or report.get("final_error_flags") != 0:
        raise ValueError("event report must pass exact comparison with zero FPGA errors")
    if channels < 1 or pre < 0 or post < 0 or lsb_uv <= 0:
        raise ValueError("invalid channel count, window, or ADC scale")
    if positions is not None and positions.shape != (channels, 3):
        raise ValueError(f"positions must have shape ({channels}, 3)")
    sample_words = adc_path.stat().st_size // 2
    if adc_path.stat().st_size % 2 or sample_words % channels:
        raise ValueError("ADC file does not contain complete signed-i16 frames")
    frames = sample_words // channels
    if report.get("samples") != sample_words:
        raise ValueError("hardware report sample count differs from ADC file")
    source = np.memmap(adc_path, dtype="<i2", mode="r", shape=(frames, channels))
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict] = []
    edge_inputs: list[StreamEvent] = []
    for event in report["events"]:
        if event.get("status_flags", 0):
            raise ValueError(f"event {event['seq']} has nonzero status flags")
        sample, channel, seq = int(event["sample"]), int(event["channel"]), int(event["seq"])
        if not 0 <= sample < frames or not 0 <= channel < channels:
            raise ValueError(f"event {seq} is outside the ADC recording")
        start, stop = max(0, sample - pre), min(frames, sample + post + 1)
        codes = np.array(source[start:stop], copy=True)
        arrays[f"event_{seq}_codes"] = codes
        metadata.append({"event_id": seq, "sample": sample, "channel": channel,
                         "start_sample": start, "stop_sample": stop,
                         "shape": list(codes.shape), "source": "host_adc"})
        if positions is not None:
            snippet = Snippet(event_id=seq, t_sample=sample, t0_sample=start,
                              sample_rate_hz=32_000.0,
                              channel_ids=tuple(range(channels)),
                              voltages_uv=codes.astype(np.float64) * lsb_uv,
                              electrode_xyz_um=positions,
                              peak_channel=channel)
            edge_inputs.append(StreamEvent(seq, sample, channel, snippet=snippet))
    result = {"session": report["session"], "adc": str(adc_path),
              "sample_rate_hz": 32_000, "lsb_uv": lsb_uv,
              "snippet_source": "host_adc", "snippets": metadata}
    if positions is not None:
        assembled = ingest(edge_inputs, session_id=str(report["session"]))
        localized = localize_events(assembled)
        result["edge_ingested"] = len(assembled)
        result["localization"] = [{"event_id": e.event_id,
                                   "available": bool(e.localization and e.localization.available),
                                   "xyz_um": list(e.localization.xyz_um)
                                   if e.localization and e.localization.xyz_um else None}
                                  for e in localized]
    return arrays, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware_json", type=Path)
    parser.add_argument("adc_i16le", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--pre", type=int, default=16, help="samples before event")
    parser.add_argument("--post", type=int, default=32, help="samples after event")
    parser.add_argument("--lsb-uv", type=float, required=True)
    parser.add_argument("--positions-npy", type=Path, help="[channels,3] electrode xyz in um")
    args = parser.parse_args()
    report = json.loads(args.hardware_json.read_text())
    positions = np.load(args.positions_npy) if args.positions_npy else None
    arrays, summary = retrieve(report, args.adc_i16le, channels=args.channels,
                               pre=args.pre, post=args.post,
                               lsb_uv=args.lsb_uv, positions=positions)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(arrays)} host ADC snippets to {args.output_npz}")


if __name__ == "__main__":
    main()
