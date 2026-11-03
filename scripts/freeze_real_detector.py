"""Write the frozen real-recording detector contract and RTL non-follow decision."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.real_detector import (
    FROZEN_REAL_DETECTOR_NAME,
    FROZEN_REAL_DETECTOR_RATIONALE,
    frozen_real_detector,
)


def main() -> None:
    cfg = frozen_real_detector(30_000.0)
    report = {
        "name": FROZEN_REAL_DETECTOR_NAME,
        "config": asdict(cfg),
        "rationale": FROZEN_REAL_DETECTOR_RATIONALE,
        "selection_evidence": "reports/ibl-detection-localization.json detector_variants at 1 ms vs sorting",
        "rtl_follows": False,
        "rtl_reason": (
            "Median CAR is required before the frozen 4σ comparison and is not present "
            "in rtl/neurosig_pipeline.sv. Changing only the offline threshold registers "
            "without CAR would not reproduce this Python detector."
        ),
        "hardware_note": "A later RTL revision could add a median-CAR stage, then load 4σ MAD registers.",
    }
    path = Path("reports/frozen-real-detector.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
