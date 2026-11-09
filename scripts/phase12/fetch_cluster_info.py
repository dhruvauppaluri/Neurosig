"""Fetch IBL sorter per-cluster depth (clusters.depths, amplitude-weighted centre of mass along the probe, um) and
quality labels for every Phase 5-12 insertion (data/ibl-venv). Writes data/phase12/cluster_info.pkl:
{session: {"cluster_id": [...], "depth_um": [...], "label": [...], "firing_rate": [...]}} as plain lists.
Used only as a second, independent *proxy* of unit depth (it is itself a sorter-derived estimate, not ground truth)."""
import json, pickle, socket, sys
from pathlib import Path
import numpy as np
socket.setdefaulttimeout(120)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from one.api import ONE
from edge.ibl import SORTING_REVISION

sess = {}
for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json",
          "reports/phase10/split_manifest_phase10.json", "reports/phase11/split_manifest_phase11.json",
          "reports/phase12/split_manifest_phase12.json"):
    p = ROOT / f
    if p.exists():
        sess.update({k: v["pid"] for k, v in json.loads(p.read_text())["sessions"].items()})
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True,
          cache_dir="data/neuropixels/ibl_cache")
out_p = ROOT / "data/phase12/cluster_info.pkl"
out = pickle.load(open(out_p, "rb")) if out_p.exists() else {}
for k, pid in sorted(sess.items()):
    if k in out:
        continue
    try:
        eid, probe = one.pid2eid(pid)
        col = f"alf/{probe}/pykilosort"
        dep = one.load_dataset(eid, "clusters.depths.npy", collection=col, revision=SORTING_REVISION)
        met = one.load_dataset(eid, "clusters.metrics.pqt", collection=col, revision=SORTING_REVISION)
        ids = np.asarray(met["cluster_id"]) if "cluster_id" in met.columns else np.arange(len(dep))
        out[k] = {"pid": pid, "cluster_id": ids.astype(int).tolist(), "depth_um": np.asarray(dep, float)[ids].tolist(),
                  "label": np.asarray(met["label"], float).tolist(), "firing_rate": np.asarray(met["firing_rate"], float).tolist()}
        print(k, len(dep), flush=True)
    except Exception as e:  # noqa: BLE001
        print(k, "failed", type(e).__name__, str(e)[:100], flush=True)
out_p.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(out, open(out_p, "wb"))
