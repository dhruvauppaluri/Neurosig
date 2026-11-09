"""Phase 13: build compact IBL caches for the 19 Phase 5-12 insertions A-S (data/ibl-venv) and write the dataset
inventory reports/phase13/ibl_inventory.json (trial counts, contrast composition, visual units, hemisphere)."""
import json, socket, sys, time
from pathlib import Path
import numpy as np
socket.setdefaulttimeout(180)
sys.path.insert(0, ".")
from edge.phase13 import ibl_data as D
sess = {}
for f in ("reports/phase5/split_manifest.json", "reports/phase6/split_manifest_phase6.json", "reports/phase10/split_manifest_phase10.json",
          "reports/phase11/split_manifest_phase11.json", "reports/phase12/split_manifest_phase12.json"):
    sess.update({k: v["pid"] for k, v in json.loads(Path(f).read_text())["sessions"].items()})
only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
inv_p = Path("reports/phase13/ibl_inventory.json")
inv = json.loads(inv_p.read_text()) if inv_p.exists() else {}
for k, pid in sorted(sess.items()):
    if only and k not in only:
        continue
    t = time.time()
    try:
        D.fetch(k, pid)
        d = D.load(k)
        cl, cr = np.nan_to_num(d["trial_contrastLeft"]), np.nan_to_num(d["trial_contrastRight"])
        c = np.maximum(cl, cr)
        inv[k] = {"pid": pid, "eid": str(d["eid"]), "n_trials": int(len(cl)), "n_vis_clusters": int(len(d["vis_clusters"])),
                  "n_vis_good": int(np.sum(d["cluster_label"] >= 1)), "vis_areas": sorted(set(map(str, d["cluster_acronym"]))),
                  "n_vis_spikes": int(len(d["spike_times"])), "hemisphere_ml_um": float(d["hemisphere_ml_um"]),
                  "contrast_counts": {str(v): int(np.sum(np.isclose(c, v))) for v in (0.0, 0.0625, 0.125, 0.25, 1.0)},
                  "side_counts": {"left": int(np.sum(cl > 0)), "right": int(np.sum(cr > 0)), "zero": int(np.sum(c == 0))}}
        print(k, inv[k]["n_trials"], inv[k]["n_vis_clusters"], inv[k]["vis_areas"][:4], f"{time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        inv[k] = {"pid": pid, "status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
        print(k, "failed", e, flush=True)
    inv_p.write_text(json.dumps(inv, indent=1) + "\n")
