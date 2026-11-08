"""Fetch IBL sorter quality metrics (clusters.metrics.pqt) for the 12 previously used insertions A-L (data/ibl-venv).
Writes data/phase11/cluster_metrics.pkl: {session: DataFrame subset}. Network only; small files."""
import json, pickle, socket, sys
from pathlib import Path
socket.setdefaulttimeout(120)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from one.api import ONE
from edge.ibl import SORTING_REVISION
sess = {}
for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json", "reports/phase10/split_manifest_phase10.json"):
    sess.update({k: v["pid"] for k, v in json.loads((ROOT / f).read_text())["sessions"].items()})
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True,
          cache_dir="data/neuropixels/ibl_cache")
out = {}
for k, pid in sorted(sess.items()):
    try:
        eid, probe = one.pid2eid(pid)
        df = one.load_dataset(eid, "clusters.metrics.pqt", collection=f"alf/{probe}/pykilosort", revision=SORTING_REVISION)
        cols = [c for c in ("cluster_id", "label", "amp_median", "firing_rate", "presence_ratio", "contamination", "noise_cutoff",
                            "slidingRP_viol", "missed_spikes_est") if c in df.columns]
        out[k] = df[cols].copy(); print(k, len(df), cols, flush=True)
    except Exception as e:  # noqa: BLE001
        print(k, "failed", type(e).__name__, str(e)[:80], flush=True)
(ROOT / "data/phase11").mkdir(parents=True, exist_ok=True)
pickle.dump(out, open(ROOT / "data/phase11/cluster_metrics.pkl", "wb"))
