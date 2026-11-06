"""Lightweight search for IBL insertions with >=32 visual-cortex contacts (downloads only the 3 kB region file)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from one.api import ONE
from iblatlas.regions import BrainRegions
from edge.ibl import SORTING_REVISION
USED = {"4836a465-c691-4852-a0b1-dcd2b1ce38a1", "1a60a6e1-da99-4d4e-a734-39b1d4544fad", "7a620688-66cb-44d3-b79b-ccac1c8ba23e", "b114e7e0-92b3-43bb-9529-53860e52606d"}
PROBED_NO_VIS = {"24149f5c-2984-4b48-922b-3cd1452cd4bd", "9b5fec96-9f19-4ddf-ba1c-3175e15245e9", "a91e4882-d3aa-4339-8b14-e94fa419f328", "49c2ea3d-2b50-4e8a-b124-9e190960784e", "06b677d6-b074-4444-bd4a-3235a310e6e0"}
one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True, cache_dir="data/neuropixels/ibl_cache")
ins = one.alyx.rest("insertions", "list", django="json__extended_qc__alignment_resolved,True")
br = BrainRegions(); out = []
for i in ins:
    pid = i["id"]
    if pid in USED or pid in PROBED_NO_VIS:
        continue
    try:
        eid, probe = one.pid2eid(pid)
        ids = one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", collection=f"alf/{probe}/pykilosort", revision=SORTING_REVISION)
        ac = np.asarray(br.id2acronym(ids)).astype(str)
        rec = dict(pid=pid, lab=i.get("session_info", {}).get("lab", ""), visp=int(np.char.startswith(ac, "VISp").sum()), vis=int(np.char.startswith(ac, "VIS").sum()))
    except Exception as e:
        rec = dict(pid=pid, error=f"{type(e).__name__}: {str(e)[:60]}")
    out.append(rec); print(json.dumps(rec), flush=True)
Path("reports/phase6/insertion_search.json").write_text(json.dumps(out, indent=1))
