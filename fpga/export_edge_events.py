"""Convert verified C5G event JSON to existing NSG1 EVENT packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.link.codec import EventPacket, encode_event
from edge.pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.hardware_json.read_text())
    if not report.get("comparison", {}).get("exact_match"):
        parser.error("hardware event comparison is absent or failed")
    if report.get("final_error_flags") != 0:
        parser.error("hardware reported a transfer or overflow error")
    events = report["events"]
    if any(event.get("status_flags", 0) for event in events):
        parser.error("hardware event record has nonzero status flags")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with args.output.open("wb") as stream:
        for event in events:
            code = int(event["code"])
            if not -32768 <= code <= 32767:
                raise ValueError(f"event {event['seq']} amplitude {code} exceeds NSG1 int16")
            frame = encode_event(EventPacket(
                event_id=int(event["seq"]), t_sample=int(event["sample"]),
                channel_id=int(event["channel"]), amplitude=code,
                seq=int(event["seq"]),
            ))
            stream.write(frame)
            frames.append(frame)
    edge_events = ingest(frames, session_id=str(report["session"]))
    if [(e.event_id, e.t_sample, e.channel_id) for e in edge_events] != [
        (int(e["seq"]), int(e["sample"]), int(e["channel"])) for e in events
    ]:
        raise RuntimeError("edge event assembler changed the verified event order")
    edge_path = args.output.with_suffix(".edge.json")
    edge_path.write_text(json.dumps({
        "session": report["session"], "event_count": len(edge_events),
        "events": [{"event_id": e.event_id, "sample": e.t_sample,
                    "channel": e.channel_id} for e in edge_events],
    }, indent=2) + "\n")
    print(f"Wrote {len(events)} NSG1 EVENT frames to {args.output}; "
          f"edge ingestion summary to {edge_path}")


if __name__ == "__main__":
    main()
