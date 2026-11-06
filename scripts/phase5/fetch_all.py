"""Download/cache every manifest window (needs data/ibl-venv)."""
import socket, sys, time
socket.setdefaulttimeout(120)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase5.data import fetch_window, load_manifest
args = sys.argv[1:]
sessions = {a.split(":")[1] for a in args if a.startswith("session:")}
splits = {a for a in args if not a.startswith("session:")} or {"seen", "calibration", "selection", "heldout"}
for w in load_manifest()["windows"]:
    if w["split"] in splits and (not sessions or w["session"] in sessions):
        t = time.time()
        for attempt in range(4):
            try:
                p = fetch_window(w["pid"], w["t0_s"], w["duration_s"])
                print("ok", w["session"], w["split"], w["t0_s"], f"{time.time()-t:.0f}s", flush=True)
                break
            except Exception as e:
                print("retry", attempt, w["session"], w["split"], w["t0_s"], type(e).__name__, str(e)[:80], flush=True)
                time.sleep(10)
