#!/usr/bin/env python3
"""Run the synthetic localization benchmark and write reports/localization-benchmark.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge import SEED
from edge.benchmark import run_localization_benchmark


def main() -> int:
    report = run_localization_benchmark(seed=SEED)
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "localization-benchmark.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"seed={report['seed']}  sample_rate_hz={report['sample_rate_hz']}  channels={report['n_channels']}")
    print(
        f"snippet {report['snippet_pre_ms']} ms pre / {report['snippet_post_ms']} ms post  "
        f"estimates={report['n_available']}/{report['n_estimates']} available"
    )
    print("all conditions (noise + overlap sweep)")
    print("method            median_error_um  p90_error_um  median_latency_s")
    for name, stats in report["methods"].items():
        med = stats["median_error_um"]
        p90 = stats["p90_error_um"]
        lat = stats["median_latency_s"]
        print(f"{name:<16}  {med:14.3f}  {p90:12.3f}  {lat:16.6f}")
    print("low-noise isolated sources")
    print("method            median_error_um  p90_error_um  median_latency_s")
    for name, stats in report["methods_low_noise_isolated"].items():
        med = stats["median_error_um"]
        p90 = stats["p90_error_um"]
        lat = stats["median_latency_s"]
        print(f"{name:<16}  {med:14.3f}  {p90:12.3f}  {lat:16.6f}")
    mearec = report["mearec"]
    print(f"mearec skipped={mearec.get('skipped')}  {mearec.get('limitation', '')}")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
