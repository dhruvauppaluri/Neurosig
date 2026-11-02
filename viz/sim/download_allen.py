"""Optional helper for Allen Visual Coding Neuropixels downloads.

This script does not run during tests. It never feeds spike times into the ADC
pipeline. Full sessions belong under gitignored data/neuropixels/.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "data" / "neuropixels"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    readme = TARGET / "README.txt"
    readme.write_text(
        "Place AllenSDK-exported NWB or converted JSON fixtures here.\n"
        "Install optional extras before calling allensdk:\n"
        "  .venv/bin/python -m pip install allensdk pynwb\n"
        "Then convert to viz/fixtures schema. Sorted spikes are playback only.\n"
        "Raw voltage, if present, must go through the ADC path separately.\n"
    )
    print(f"Wrote {readme}")
    print("AllenSDK is optional and is not required for the visualization smoke test.")


if __name__ == "__main__":
    main()
