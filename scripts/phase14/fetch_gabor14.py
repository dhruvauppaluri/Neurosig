"""Phase 14V supplement (data/allen-venv): receptive-field-mapping Gabor presentations (stimulus POSITION labels) for
every session already cached by fetch_allen14.py. Spike counts of the same good VIS units (same order as the main
cache) in 25 ms bins over 0-250 ms, plus x_position, y_position, orientation per presentation.
Writes data/phase14/allen_gabor_<session>.npz. Usage: fetch_gabor14.py"""
import json, sys, time
from pathlib import Path
import h5py, numpy as np, remfile

sys.path.insert(0, ".")
from scripts.phase14.fetch_allen14 import ASSETS, INV, sess_of, strs, num
from edge.phase14.allen_map import bin_counts

OUT = Path("data/phase14")
inv = json.loads(INV.read_text())
by = {sess_of(a): a for a in ASSETS}
for s, r in inv.items():
    if r.get("status") != "ok" or (OUT / f"allen_gabor_{s}.npz").exists():
        continue
    t = time.time()
    try:
        rf = remfile.File(f"https://api.dandiarchive.org/api/assets/{by[s]['asset_id']}/download/", _min_chunk_size=4 << 20,
                          _max_threads=8, _max_chunk_size=64 << 20)
        with h5py.File(rf, "r") as f:
            iv = f["intervals"]
            if "gabors_presentations" not in iv:
                print(s, "no gabors", flush=True); continue
            g = iv["gabors_presentations"]
            on = g["start_time"][:]; x = num(g["x_position"]); y = num(g["y_position"]); ori = num(g["orientation"])
            ids = np.load(OUT / f"allen_{s}.npz")["unit_id"]
            u = f["units"]; uid = u["id"][:]; pos = {int(v): i for i, v in enumerate(uid)}
            st_index = u["spike_times_index"][:]; starts = np.concatenate([[0], st_index[:-1]])
            C = np.zeros((len(on), len(ids), 10), np.uint8)
            st = u["spike_times"]
            for j, v in enumerate(ids):
                i = pos[int(v)]
                sp = st[int(starts[i]):int(st_index[i])]
                C[:, j] = np.minimum(bin_counts(sp, on, 10, 0.025), 255)
            np.savez_compressed(OUT / f"allen_gabor_{s}.npz", counts=C, start=on, x=x, y=y, orientation=ori)
            print(s, "ok", len(on), sorted(set(np.round(x[np.isfinite(x)]).tolist()))[:10], f"{time.time() - t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(s, "failed", type(e).__name__, str(e)[:100], flush=True)
