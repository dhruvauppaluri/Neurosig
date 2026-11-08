"""Search for NEW IBL insertions (never used in Phases 3-10) with >=32 visual-cortex contacts and raw AP data.

Downloads only the small region file per insertion. Stops after ``--max-found`` hits. Run with data/ibl-venv.
Output: reports/phase11/insertion_search.json (availability facts only; no voltage is looked at).
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

USED = {  # Phases 3-10 (A-H) plus insertions probed earlier and found unsuitable
    "4836a465-c691-4852-a0b1-dcd2b1ce38a1", "1a60a6e1-da99-4d4e-a734-39b1d4544fad", "7a620688-66cb-44d3-b79b-ccac1c8ba23e",
    "b114e7e0-92b3-43bb-9529-53860e52606d", "c5b9e063-f640-4936-b851-f7602cb6659b", "5135e93f-2f1f-4301-9532-b5ad62548c49",
    "18d316bf-d322-4c5c-814e-a58147f7bf5f", "730770d6-617a-4ada-95db-a48521befda5", "366c7c11-0000-0000-0000-000000000000",
    "24149f5c-2984-4b48-922b-3cd1452cd4bd", "9b5fec96-9f19-4ddf-ba1c-3175e15245e9", "a91e4882-d3aa-4339-8b14-e94fa419f328",
    "49c2ea3d-2b50-4e8a-b124-9e190960784e", "06b677d6-b074-4444-bd4a-3235a310e6e0",
    # Phase 10 locked I-L and the two skipped degenerate-sync candidates
    "0851db85-2889-4070-ac18-a40e8ebd96ba", "0a0184b3-9e1a-4c36-98f4-00ae3beb8f01", "143dd7cf-6a47-47a1-906d-927ad7fe9117",
    "02cc03e4-8015-4050-bb42-6c832091febb", "0393f34c-a2bd-4c01-99c9-f6b4ec6e786d", "071f02e7-752a-4094-af79-8dd764e9d85d"}
USED_LABS = {"hausserlab", "angelakilab", "steinmetzlab", "churchlandlab_ucla", "wittenlab", "cortexlab", "mrsicflogellab"}

ap = argparse.ArgumentParser()
ap.add_argument("--max-found", type=int, default=8)
ap.add_argument("--max-probe", type=int, default=120)
a = ap.parse_args()
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international",
          silent=True, cache_dir="data/neuropixels/ibl_cache")
ins = one.alyx.rest("insertions", "list", django="json__extended_qc__alignment_resolved,True")
ins = sorted(ins, key=lambda i: i["id"])           # deterministic order
br = BrainRegions(); out, found, labs = [], 0, set()
for i in ins[: a.max_probe * 4]:
    pid = i["id"]
    lab = (i.get("session_info") or {}).get("lab", "")
    if pid in USED or str(pid)[:8] == "366c7c11" or lab in labs:
        continue
    try:
        eid, probe = one.pid2eid(pid)
        ids = one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", collection=f"alf/{probe}/pykilosort",
                               revision=SORTING_REVISION)
        ac = np.asarray(br.id2acronym(ids)).astype(str)
        vis = int(np.char.startswith(ac, "VIS").sum())
        raw = [d for d in one.list_datasets(eid, collection=f"raw_ephys_data/{probe}") if d.endswith(".ap.cbin") or d.endswith(".ap.ch")]
        rec = dict(pid=pid, lab=lab, vis=vis, visp=int(np.char.startswith(ac, "VISp").sum()), raw_ap=len(raw) >= 2,
                   new_lab=lab not in USED_LABS)
        if vis >= 32 and rec["raw_ap"]:
            found += 1; labs.add(lab)
    except Exception as e:  # noqa: BLE001
        rec = dict(pid=pid, lab=lab, error=f"{type(e).__name__}: {str(e)[:60]}")
    out.append(rec); print(json.dumps(rec), flush=True)
    (ROOT / "reports/phase11").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/phase11/insertion_search.json").write_text(json.dumps(out, indent=1) + "\n")
    if found >= a.max_found or len(out) >= a.max_probe:
        break
print("found", found)
