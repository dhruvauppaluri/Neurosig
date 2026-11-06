"""The committed RTL test vectors must equal a fresh deterministic regeneration (and were oracle-verified when written)."""
import json
from pathlib import Path

from scripts.phase6.generate_rtl_vectors import all_files, digest

ROOT = Path(__file__).resolve().parents[1]


def test_committed_vectors_match_regeneration():
    manifest = json.loads((ROOT / "tests" / "vectors" / "phase6" / "manifest.json").read_text())["files"]
    fresh = {k: digest(v) for k, v in all_files().items()}
    assert fresh == manifest


def test_committed_vector_files_match_manifest_hashes():
    base = ROOT / "tests" / "vectors" / "phase6"
    manifest = json.loads((base / "manifest.json").read_text())["files"]
    for name, h in manifest.items():
        assert digest((base / name).read_text().rstrip("\n").split("\n")) == h, name
