"""Run the synthetic spatial vs count decoder comparison and write a JSON report."""

from __future__ import annotations

import os

for _thread_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.decode.experiment import NEUROPIXELS_ZIP, build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/edge-decode.json"))
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--neuropixels-zip", type=Path, default=NEUROPIXELS_ZIP)
    args = parser.parse_args()
    report = build_report(seed=args.seed, zip_path=args.neuropixels_zip)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    baseline = report["baseline"]["test_accuracy"]
    spatial = report["spatial"]["test_accuracy"]
    improved = report["spatial_improved"]
    npx = report["neuropixels"]["status"]
    print(
        f"baseline={baseline:.6f} spatial={spatial:.6f} "
        f"spatial_improved={improved} neuropixels={npx} -> {args.report}"
    )


if __name__ == "__main__":
    main()
