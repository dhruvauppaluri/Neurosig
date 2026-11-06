"""Regenerate reports/phase5/detector-v2.json (the frozen Phase 5 configs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase5.detectors import BASELINE

V2 = BASELINE.replace(highpass_hz=200.0, lowpass_hz=4500.0)
V2_MEANCAR = V2.replace(car="mean")
if __name__ == "__main__":
    p = Path(__file__).resolve().parents[2] / "reports" / "phase5" / "detector-v2.json"
    cur = json.loads(p.read_text())
    assert cur["configs"] == {"baseline": BASELINE.as_dict(), "v2": V2.as_dict(), "v2_meancar": V2_MEANCAR.as_dict()}, "frozen file differs"
    print("detector-v2.json matches the code definitions")
