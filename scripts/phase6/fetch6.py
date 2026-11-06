"""Download/cache the Phase 6 locked-final windows (needs data/ibl-venv). Idempotent; retries."""
import json, socket, sys, time
from pathlib import Path
socket.setdefaulttimeout(120)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.phase5.data import fetch_window
man = json.loads((Path(__file__).resolve().parents[2] / "reports/phase6/split_manifest_phase6.json").read_text())
sess = {a.split(":")[1] for a in sys.argv[1:] if a.startswith("session:")}
for w in man["windows"]:
    if sess and w["session"] not in sess:
        continue
    for attempt in range(4):
        t = time.time()
        try:
            fetch_window(w["pid"], w["t0_s"], w["duration_s"])
            print("ok", w["session"], w["t0_s"], f"{time.time()-t:.0f}s", flush=True); break
        except Exception as e:
            print("retry", attempt, w["session"], w["t0_s"], type(e).__name__, str(e)[:80], flush=True); time.sleep(10)
