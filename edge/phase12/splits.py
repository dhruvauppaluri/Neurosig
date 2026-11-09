"""Phase 12 data splits (session-disjoint).

dev     Phase 5 calibration + selection windows of sessions A-C (development decisions)
select  sessions D (Phase 5 held-out), E-H (Phase 6), I-L (Phase 10), M-P (Phase 11): all previously used, read only
        for model selection / generalisation checks. Phase 5 held-out windows of A-C are excluded so that select is
        session-disjoint from dev.
final   sessions Q-S, reports/phase12/split_manifest_phase12.json (fresh; evaluated once by run_final12.py)
"""
import json
from pathlib import Path

from edge.phase9 import harness as H

ROOT = Path(__file__).resolve().parents[2]


def split_windows(split: str) -> list[dict]:
    if split == "dev":
        return H.manifest_windows("dev")
    if split == "select":
        m11 = json.loads((ROOT / "reports/phase11/split_manifest_phase11.json").read_text())["windows"]
        w = [x for x in H.manifest_windows("heldout5") if x["session"] == "D"]
        return w + H.manifest_windows("locked") + H.manifest_windows("locked10") + [dict(x) for x in m11]
    if split == "final":
        return [dict(x) for x in json.loads((ROOT / "reports/phase12/split_manifest_phase12.json").read_text())["windows"]]
    raise ValueError(split)
