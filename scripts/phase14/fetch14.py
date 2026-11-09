"""Download the Phase 14 final data (data/ibl-venv): raw AP windows of the manifest, the compact visual-decoding cache
(edge/phase13/ibl_data.fetch, keys T-W) and sorter cluster metadata (depth, label, rate, plus the quality metrics
used to stratify online-grouping results). Idempotent; retries; failures reported, never silently skipped.
Downloading does not evaluate anything: no Phase 14 model touches these sessions before run_final14.py."""
import json, pickle, socket, sys, time
from pathlib import Path
socket.setdefaulttimeout(180)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.phase5.data import fetch_window

man = json.loads((ROOT / "reports/phase14/split_manifest_phase14.json").read_text())
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

from one.api import ONE
from edge.ibl import SORTING_REVISION
from edge.phase13 import ibl_data as D
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True,
          cache_dir="data/neuropixels/ibl_cache")
ci_p = ROOT / "data/phase14/cluster_info14.pkl"
ci = pickle.load(open(ci_p, "rb")) if ci_p.exists() else {}
for k, v in man["sessions"].items():
    for attempt in range(3):
        try:
            D.fetch(k, v["pid"]); print("decode cache ok", k, flush=True); break
        except Exception as e:  # noqa: BLE001
            print("decode cache retry", k, type(e).__name__, str(e)[:80], flush=True); time.sleep(10)
    if k in ci:
        continue
    try:
        eid, probe = one.pid2eid(v["pid"])
        col = f"alf/{probe}/pykilosort"
        dep = one.load_dataset(eid, "clusters.depths.npy", collection=col, revision=SORTING_REVISION)
        met = one.load_dataset(eid, "clusters.metrics.pqt", collection=col, revision=SORTING_REVISION)
        ids = np.asarray(met["cluster_id"]) if "cluster_id" in met.columns else np.arange(len(dep))
        ci[k] = {"pid": v["pid"], "cluster_id": ids.astype(int).tolist(), "depth_um": np.asarray(dep, float)[ids].tolist(),
                 "label": np.asarray(met["label"], float).tolist(), "firing_rate": np.asarray(met["firing_rate"], float).tolist()}
        print("cluster info ok", k, len(dep), flush=True)
    except Exception as e:  # noqa: BLE001
        print("cluster info failed", k, type(e).__name__, str(e)[:100], flush=True)
pickle.dump(ci, open(ci_p, "wb"))
print("done")
