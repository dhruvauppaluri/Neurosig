"""Phase 15: search for IBL insertions never used in Phases 3-13 (>= 32 visual-cortex contacts, raw AP present).

The Phase 11 search stopped after probing 113 insertions and skipped repeated labs, so its eligible list was exhausted
by Phase 12. This search continues in the same deterministic order (alignment-resolved insertions sorted by id),
skips every pid already probed or used, and allows repeated labs (different animals / sessions); distinct labs are
preferred later by the manifest. Only small region files are downloaded; no voltage is looked at.
Output: reports/phase15/insertion_search15.json. Run with data/ibl-venv.
"""
import argparse, json, socket, sys
from pathlib import Path
socket.setdefaulttimeout(120)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from one.api import ONE
from iblatlas.regions import BrainRegions
from edge.ibl import SORTING_REVISION

OUT = ROOT / "reports/phase15/insertion_search15.json"
probed = {c["pid"] for c in json.loads((ROOT / "reports/phase11/insertion_search.json").read_text())} | {c["pid"] for c in json.loads((ROOT / "reports/phase14/insertion_search14.json").read_text())}
used = set()
for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json",
          "reports/phase10/split_manifest_phase10.json", "reports/phase11/split_manifest_phase11.json",
          "reports/phase12/split_manifest_phase12.json", "reports/phase14/split_manifest_phase14.json"):
    for v in json.loads((ROOT / f).read_text())["sessions"].values():
        used.add(v["pid"])
ap = argparse.ArgumentParser()
ap.add_argument("--max-found", type=int, default=10)
ap.add_argument("--max-probe", type=int, default=200)
a = ap.parse_args()
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
          silent=True, cache_dir="data/neuropixels/ibl_cache")
ins = sorted(one.alyx.rest("insertions", "list", django="json__extended_qc__alignment_resolved,True"), key=lambda i: i["id"])
br = BrainRegions()
out = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["pid"] for r in out}
found = sum(1 for r in out if r.get("vis", 0) >= 32 and r.get("raw_ap"))
for i in ins:
    pid = i["id"]
    if pid in probed or pid in used or pid in done:
        continue
    lab = (i.get("session_info") or {}).get("lab", "")
    try:
        eid, probe = one.pid2eid(pid)
        ids = one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", collection=f"alf/{probe}/pykilosort",
                               revision=SORTING_REVISION)
        ac = np.asarray(br.id2acronym(ids)).astype(str)
        vis = int(np.char.startswith(ac, "VIS").sum())
        raw = [d for d in one.list_datasets(eid, collection=f"raw_ephys_data/{probe}") if d.endswith(".ap.cbin") or d.endswith(".ap.ch")]
        rec = dict(pid=pid, lab=lab, vis=vis, visp=int(np.char.startswith(ac, "VISp").sum()), raw_ap=len(raw) >= 2)
        if vis >= 32 and rec["raw_ap"]:
            found += 1
    except Exception as e:  # noqa: BLE001
        rec = dict(pid=pid, lab=lab, error=f"{type(e).__name__}: {str(e)[:60]}")
    out.append(rec); print(json.dumps(rec), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    if found >= a.max_found or len(out) >= a.max_probe:
        break
print("found", found)
