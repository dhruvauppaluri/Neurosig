"""Collect measured Quartus and C5G host-run numbers without filling gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def read_lines(path: Path, patterns: tuple[str, ...]) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(errors="replace").splitlines()
            if any(re.search(pattern, line, re.I) for pattern in patterns)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--hardware-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.build_dir / "neurosig_c5g"
    report: dict = {
        "fit_summary": read_lines(prefix.with_suffix(".fit.summary"),
                                  (r"logic utilization", r"total registers", r"total block memory bits",
                                   r"total ram blocks", r"total dsp blocks")),
        "timing_lines": read_lines(prefix.with_suffix(".sta.rpt"),
                                   (r";\s*\d+\.\d+ MHz\s*;[^\n]*CLOCK_50_B5B",
                                    r"worst-case setup slack")),
        "setup_summary": read_lines(prefix.with_suffix(".sta.summary"),
                                    (r"^Type\s*:.*Setup 'CLOCK_50_B5B'", r"^Slack\s*:")),
        "vectorless_power_estimate": read_lines(prefix.with_suffix(".pow.rpt"),
                                                 (r"^;\s*(Total Thermal Power Dissipation|"
                                                  r"Core Dynamic Thermal Power Dissipation|"
                                                  r"Core Static Thermal Power Dissipation|"
                                                  r"I/O Thermal Power Dissipation|"
                                                  r"Power Estimation Confidence)\s*;",)),
        "source_files": {suffix: str(prefix.with_suffix(suffix)) for suffix in
                         (".fit.summary", ".sta.rpt", ".sta.summary", ".pow.rpt")},
    }
    if args.hardware_json:
        hardware = json.loads(args.hardware_json.read_text())
        report["physical_host_run"] = {
            key: hardware[key] for key in
            ("samples", "frames", "host_wall_seconds", "host_samples_per_second",
             "fifo_peak_observed", "fifo_peak_hardware", "feed_cycles",
             "feed_cycles_per_sample", "check_cycles", "emitted_events_hardware",
             "final_error_flags", "comparison", "verilator_comparison")
            if key in hardware
        }
        if hardware.get("feed_cycles_per_sample"):
            report["physical_host_run"]["on_chip_feed_samples_per_second_at_50mhz"] = (
                50_000_000 / hardware["feed_cycles_per_sample"]
            )
        if hardware.get("host_wall_seconds") and hardware.get("events") is not None:
            report["physical_host_run"]["captured_events_per_host_second"] = (
                len(hardware["events"]) / hardware["host_wall_seconds"]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
