"""Write the Phase 5 split manifest BEFORE any tuning (seeded, disjoint windows).

Sessions A,B,C contribute calibration/selection/heldout windows; session D is
entirely held out (cross-session generalization). Previously used IBL windows
are listed as 'seen' and never used for selection or the final claim.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase5.data import MANIFEST

SESSIONS = {
    "A": ("4836a465-c691-4852-a0b1-dcd2b1ce38a1", 5000.0, "seen_before"),
    "B": ("1a60a6e1-da99-4d4e-a734-39b1d4544fad", 4000.0, "seen_before"),
    "C": ("7a620688-66cb-44d3-b79b-ccac1c8ba23e", 5014.0, "new"),
    "D": ("b114e7e0-92b3-43bb-9529-53860e52606d", 4097.0, "new"),
}
SEEN = {"A": [50.0, 100.0, 200.0], "B": [50.0, 90.0]}
DUR = 2.0
SEED = 20261105

def main():
    rng = np.random.default_rng(SEED)
    windows = []
    for key, (pid, total, _) in SESSIONS.items():
        for t in SEEN.get(key, []):
            windows.append(dict(session=key, pid=pid, t0_s=t, duration_s=1.0, split="seen"))
        plan = ["heldout"] * 8 if key == "D" else ["calibration"] * 4 + ["selection"] * 4 + ["heldout"] * 4
        # distinct 60 s slots between 300 s and total-120 s, random order
        slots = np.arange(300.0, total - 120.0, 60.0)
        pick = rng.choice(len(slots), size=len(plan), replace=False)
        for split, i in zip(plan, pick):
            t0 = float(slots[i] + rng.uniform(0, 30.0))
            windows.append(dict(session=key, pid=pid, t0_s=round(t0, 3), duration_s=DUR, split=split))
    man = dict(seed=SEED, window_s=DUR,
               rule="calibration=diagnostics only; selection=choose variants; heldout=evaluated once at the end; seen=reproduction only",
               sessions={k: dict(pid=v[0]) for k, v in SESSIONS.items()}, windows=windows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise SystemExit("manifest exists; refusing to overwrite (split must stay fixed)")
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    print(len(windows), "windows")
main()
