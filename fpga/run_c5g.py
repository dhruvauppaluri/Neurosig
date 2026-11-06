"""Play signed i16 sample-major ADC codes through the C5G JTAG mailbox."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TCL = Path(__file__).with_name("system_console_run.tcl")
DEFAULT_CONSOLE = Path(r"C:\altera_lite\25.1std\quartus\sopc_builder\bin\system-console.exe")


def windows_path(path: Path) -> str:
    path = path.resolve()
    if os.name == "nt":
        return str(path)
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def tcl_quote(value: str) -> str:
    return "{" + value.replace("\\", "/").replace("}", "\\}") + "}"


def pack_samples(source: Path, dest: Path, channels: int = 32) -> tuple[int, int]:
    raw = source.read_bytes()
    if len(raw) % (2 * channels):
        raise ValueError("ADC file must contain complete sample-major frames")
    count = len(raw) // 2
    with dest.open("wb") as target:
        for offset in range(0, count, 65536):
            segment = raw[offset * 2 : min(count, offset + 65536) * 2]
            values = struct.unpack(f"<{len(segment)//2}h", segment)
            target.write(struct.pack(f"<{len(values)}I", *(
                (int(code) & 0xffff) | (((offset + i) % channels) << 16)
                for i, code in enumerate(values)
            )))
    return count, count // channels


def read_events(path: Path) -> list[dict[str, int]]:
    events: dict[int, dict[str, int]] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) not in (4, 5):
                raise ValueError(f"invalid event row {row}")
            seq, index, channel, code = map(int, row[:4])
            flags = int(row[4]) if len(row) == 5 else 0
            event = {"seq": seq, "sample": index, "channel": channel,
                     "code": code, "status_flags": flags}
            if seq in events and events[seq] != event:
                raise ValueError(f"conflicting event sequence {seq}")
            events[seq] = event
    result = [events[k] for k in sorted(events)]
    if [e["seq"] for e in result] != list(range(len(result))):
        raise ValueError("missing event sequence")
    return result


def compare(events: list[dict[str, int]], expected_path: Path) -> dict:
    expected = json.loads(expected_path.read_text())["events"]
    actual = [(e["sample"], e["channel"], e["code"]) for e in events]
    reference = [(e["sample"], e["channel"], e["code"]) for e in expected]
    return {"exact_match": actual == reference, "actual_count": len(actual),
            "expected_count": len(reference), "missing": [list(e) for e in reference if e not in actual],
            "extra": [list(e) for e in actual if e not in reference]}


def compare_verilator(events: list[dict[str, int]], observed_path: Path) -> dict:
    reference = [tuple(map(int, line.split())) for line in observed_path.read_text().splitlines() if line.strip()]
    actual = [(e["sample"], e["channel"], e["code"]) for e in events]
    return {"exact_match": actual == reference, "actual_count": len(actual),
            "reference_count": len(reference)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adc", type=Path, required=True, help="little-endian signed i16, frame-major")
    parser.add_argument("--thresholds", type=Path, required=True, help="32 lines of 24-bit hex")
    parser.add_argument("--expected", type=Path, default=ROOT / "reports/rtl-pipeline-validation.json")
    parser.add_argument("--verilator-events", type=Path, help="optional observed triples from the full Verilator test")
    parser.add_argument("--skip-compare", action="store_true", help="for diagnostic vectors without a stored reference")
    parser.add_argument("--output", type=Path, default=Path("hardware-events.json"))
    parser.add_argument("--console", type=Path,
                        default=DEFAULT_CONSOLE if os.name == "nt" else
                        Path("/mnt/c/altera_lite/25.1std/quartus/sopc_builder/bin/system-console.exe"))
    parser.add_argument("--session", type=int, default=20261103)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.console.is_file():
        parser.error(f"System Console not found: {args.console}")
    if args.verilator_events and not args.verilator_events.is_file():
        parser.error(f"Verilator event file not found: {args.verilator_events}")
    thresholds = args.thresholds.read_text().splitlines()
    if len(thresholds) != 32 or any(int(x, 16) > 0xffffff for x in thresholds):
        parser.error("threshold file must have exactly 32 unsigned 24-bit hex codes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    if not args.resume:
        csv_path.write_text("")
    start = time.monotonic()
    temp_parent = None if os.name == "nt" else Path("/mnt/c/Users/dhruv/AppData/Local/Temp")
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp:
        words = Path(temp) / "words.bin"
        samples, frames = pack_samples(args.adc, words)
        runner = Path(temp) / "run.tcl"
        settings = {
            "NSG_WORDS_BIN": windows_path(words),
            "NSG_EVENTS_CSV": windows_path(csv_path),
            "NSG_THRESHOLDS": windows_path(args.thresholds),
            "NSG_SESSION": str(args.session),
            "NSG_RESUME": "1" if args.resume else "0",
        }
        runner.write_text("\n".join(
            f"set ::env({key}) {tcl_quote(value)}" for key, value in settings.items()
        ) + f"\nsource {tcl_quote(windows_path(TCL))}\n")
        result = subprocess.run([str(args.console), f"--script={windows_path(runner)}"],
                                text=True, capture_output=True)
    elapsed = time.monotonic() - start
    if result.returncode or "NSG_SUCCESS" not in result.stdout:
        raise RuntimeError(f"System Console failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    import re
    summary_match = re.search(r"NSG_SUCCESS samples=(\d+) max_fifo=(\d+) errors=(\d+)", result.stdout)
    if not summary_match:
        raise RuntimeError(f"System Console returned an incomplete summary:\n{result.stdout}")
    _, fifo_peak, error_flags = map(int, summary_match.groups())
    counters_match = re.search(
        r"feed_cycles=(\d+) check_cycles=(\d+) fifo_peak_hw=(\d+) emitted=(\d+)",
        result.stdout,
    )
    counters = tuple(map(int, counters_match.groups())) if counters_match else None
    events = read_events(csv_path)
    comparison = compare(events, args.expected) if not args.skip_compare else None
    verilator_comparison = compare_verilator(events, args.verilator_events) if args.verilator_events else None
    report = {"source": str(args.adc), "samples": samples, "frames": frames,
              "session": args.session, "events": events, "comparison": comparison,
              "verilator_comparison": verilator_comparison,
              "host_wall_seconds": elapsed,
              "host_samples_per_second": None if args.resume else samples / elapsed,
              "resumed": args.resume,
              "fifo_peak_observed": fifo_peak, "final_error_flags": error_flags,
              "feed_cycles": counters[0] if counters else None,
              "check_cycles": counters[1] if counters else None,
              "fifo_peak_hardware": counters[2] if counters else None,
              "emitted_events_hardware": counters[3] if counters else None,
              "feed_cycles_per_sample": counters[0] / samples if counters and samples else None,
              "console_output": result.stdout.strip()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))
    if comparison is not None and not comparison["exact_match"]:
        raise SystemExit("hardware events do not exactly match the reference")
    if verilator_comparison is not None and not verilator_comparison["exact_match"]:
        raise SystemExit("hardware events do not exactly match Verilator")


if __name__ == "__main__":
    main()
