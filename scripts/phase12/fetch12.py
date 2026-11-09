"""Download/cache the Phase 12 final windows (needs data/ibl-venv). Idempotent; retries; reports failures."""
import json, socket, sys, time
from pathlib import Path
socket.setdefaulttimeout(120)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.phase5.data import fetch_window
man = json.loads((ROOT / "reports/phase12/split_manifest_phase12.json").read_text())
for w in man["windows"]:
    for attempt in range(4):
        t = time.time()
        try:
            fetch_window(w["pid"], w["t0_s"], w["duration_s"])
            print("ok", w["session"], w["t0_s"], f"{time.time()-t:.0f}s", flush=True); break
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, w["session"], w["t0_s"], type(e).__name__, str(e)[:80], flush=True); time.sleep(10)
    else:
        print("FAILED", w["session"], w["t0_s"], flush=True)
