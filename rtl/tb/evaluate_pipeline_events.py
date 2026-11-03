"""Evaluate RTL merged events against MEArec ground truth after simulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.detect_spikes import DetectorConfig, Event, load_ground_truth, match_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, default=Path("/tmp/neurosig-rtl-events.txt"))
    parser.add_argument("--recording", type=Path, default=Path("data/unfiltered/recording.h5"))
    parser.add_argument("--output", type=Path, default=Path("reports/rtl-pipeline-validation.json"))
    args = parser.parse_args()
    config = DetectorConfig()
    rows = []
    for line in args.observed.read_text().splitlines():
        index, channel, code = map(int, line.split())
        rows.append({"sample": index, "channel": channel, "code": code})
    events = [Event(row["sample"], row["channel"], 0.0, 0.0) for row in rows]
    matches, metrics = match_events(events, load_ground_truth(args.recording), config)
    report = {
        "source": "full 32,000-frame by 32-channel RTL pipeline simulation",
        "simulator": subprocess.run(["verilator", "--version"], capture_output=True,
                                    text=True, check=True).stdout.strip(),
        "sample_rate_hz": 32000,
        "adc_samples": 1_024_000,
        "detector_config": asdict(config),
        "events": rows,
        "ground_truth_matches": matches,
        "ground_truth_metrics": metrics,
        "limitations": "One seeded one-second MEArec recording; thresholds calibrated offline; no physical timing or power measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RTL vs MEArec: {metrics['true_positives']} TP, "
          f"{metrics['false_positives']} FP, {metrics['false_negatives']} FN; "
          f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
