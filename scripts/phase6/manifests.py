"""Window lists for both manifests (Phase 5 splits and the Phase 6 locked-final set)."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "reports" / "phase6" / "split_manifest_phase6.json"
DETECTOR_FILE = ROOT / "reports" / "phase5" / "detector-v2.json"


def specs(split: str, which: str = "phase5"):
    if which == "phase5":
        from scripts.phase5.data import manifest_windows
        return manifest_windows(split)
    return [w for w in json.loads(P6.read_text())["windows"] if w["split"] == split] if P6.exists() else []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
