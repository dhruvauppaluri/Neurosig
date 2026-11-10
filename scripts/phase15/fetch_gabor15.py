"""Phase 15: receptive-field Gabor positions for the Phase 15 Allen mice (data/allen-venv).

Reuses the Phase 14 Gabor fetcher (scripts/phase14/fetch_gabor14.py) unchanged, pointed at the Phase 15 inventory.
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, ".")
socket.setdefaulttimeout(300)
import scripts.phase14.fetch_allen14 as F  # noqa: E402

F.INV = Path("reports/phase15/allen_inventory15.json")
src = Path("scripts/phase14/fetch_gabor14.py").read_text().replace(
    "from scripts.phase14.fetch_allen14 import ASSETS, INV, sess_of, strs, num",
    "from scripts.phase14.fetch_allen14 import ASSETS, sess_of, strs, num\nINV = F.INV")
exec(compile(src, "fetch_gabor15", "exec"), {"__name__": "__main__", "F": F})
