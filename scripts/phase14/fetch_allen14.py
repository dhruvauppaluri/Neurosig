"""Phase 14 Allen Visual Coding Neuropixels cache (DANDI 000021), run with data/allen-venv.

Fixes the Phase 13 waveform-to-electrode mapping defect. Each unit's ``waveform_mean`` block has one row per channel
of its probe, ordered by the electrode's ``local_index`` (0-383). The electrodes table is NOT stored in local_index
order, so Phase 13's "row r = r-th electrode of the probe in table order" mapping was wrong (median trough-to-peak
contact distance 86 um on the stored 32 rows; > 1 mm on full rows). Here row r maps to the electrode with
local_index r on the same probe (map_waveform_rows; regression-tested in edge/tests/test_phase14.py).

Per session: good visual units (id, area, probe, peak-channel probe position, CCF, snr, rate, quality metrics),
mean waveform on the 32 contacts nearest the peak channel with their probe positions, and spike counts in 5 ms bins
over 0-250 ms after onset for drifting gratings, static gratings and natural scenes (uint8). The natural-scene images
(stimulus templates) are stored once, downsampled 4x, if the file carries them.

Split (reserved BEFORE fetching or looking at any new session; reports/phase14/allen_split14.json):
  the 5 Phase 13 mice  -> dev (2) / select (3 = Phase 13 select + final, all already analysed)
  unused assets, seeded permutation (seed 20271431): first half -> FINAL candidates, second half -> train expansion;
  the first 6 eligible (Brain Observatory 1.1 stimuli, >= 20 good VIS units) of each list are used.
Usage: fetch_allen14.py split | fetch_allen14.py fetch [max_new_per_group]
"""
import json, sys, time
from pathlib import Path
import h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge.phase14.allen_map import bin_counts, map_waveform_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = json.loads((ROOT / "data/phase13/allen_assets.json").read_text())
P13 = ["715093703", "719161530", "721123822", "732592105", "739448407"]
OUT = ROOT / "data/phase14"; SPLIT = ROOT / "reports/phase14/allen_split14.json"; INV = ROOT / "reports/phase14/allen_inventory14.json"
BIN_S, NBIN = 0.005, 50


def sess_of(a):
    return a["path"].split("ses-")[1].split(".")[0]


def make_split():
    if SPLIT.exists():
        raise SystemExit("split exists; never redrawn")
    unused = sorted([a for a in ASSETS if sess_of(a) not in P13], key=lambda a: a["path"])
    perm = np.random.default_rng(20271431).permutation(len(unused))
    h = len(unused) // 2
    s = {"rule": "dev/select = Phase 13 mice (already analysed); final candidates = first half of a seeded permutation "
                 "of unused assets, train expansion = second half; first 6 eligible of each are used. Reserved before "
                 "any new session was opened.",
         "seed": 20271431, "dev": P13[:2], "select_p13": P13[2:],
         "final_candidates": [sess_of(unused[i]) for i in perm[:h]], "train_candidates": [sess_of(unused[i]) for i in perm[h:]]}
    SPLIT.write_text(json.dumps(s, indent=1) + "\n"); print(s)


def strs(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x[:]])


def num(x):
    v = x[:]
    if v.dtype.kind in "fiu":
        return v.astype(float)
    out = []
    for e in v:
        e = e.decode() if isinstance(e, bytes) else str(e)
        try:
            out.append(float(e))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def _process(f, sess):
    iv = f["intervals"]
    if not all(k in iv for k in ("drifting_gratings_presentations", "static_gratings_presentations", "natural_scenes_presentations")):
        return {"status": "skipped: not a Brain Observatory 1.1 session (missing DG/SG/NS)"}
    el = f["general/extracellular_ephys/electrodes"]
    eid = el["id"][:]; loc = strs(el["location"]); vpos = el["probe_vertical_position"][:]; hpos = el["probe_horizontal_position"][:]
    pid = el["probe_id"][:]; li = el["local_index"][:]
    ccf = np.stack([el[c][:] for c in ("x", "y", "z")], axis=1)
    e_ix = {int(e): i for i, e in enumerate(eid)}
    u = f["units"]
    peak = u["peak_channel_id"][:]; q = strs(u["quality"])
    ui = np.array([e_ix.get(int(p), -1) for p in peak])
    vis = np.flatnonzero((ui >= 0) & np.char.startswith(loc[np.maximum(ui, 0)], "VIS") & (q == "good"))
    if len(vis) < 20:
        return {"status": f"skipped: {len(vis)} good visual units"}
    st_index = u["spike_times_index"][:]; starts = np.concatenate([[0], st_index[:-1]])
    stim = {}
    for key, name, cols in (("dg", "drifting_gratings_presentations", ("orientation", "temporal_frequency", "contrast")),
                            ("sg", "static_gratings_presentations", ("orientation", "spatial_frequency", "phase")),
                            ("ns", "natural_scenes_presentations", ("frame",))):
        g = iv[name]
        stim[key] = {"start": g["start_time"][:], **{c: num(g[c]) for c in cols}}
    counts = {k: np.zeros((len(v["start"]), len(vis), NBIN), np.uint8) for k, v in stim.items()}
    blocks, cur = [], [vis[0]]
    for x in vis[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            blocks.append(cur); cur = [x]
    blocks.append(cur)
    st_ds = u["spike_times"]; jx = {int(x): j for j, x in enumerate(vis)}
    for b in blocks:
        lo, hi = int(starts[b[0]]), int(st_index[b[-1]])
        arr = st_ds[lo:hi]
        for x in b:
            s = arr[int(starts[x]) - lo:int(st_index[x]) - lo]
            for k, v in stim.items():
                counts[k][:, jx[int(x)]] = np.minimum(bin_counts(s, v["start"]), 255)
    wf_idx = u["waveform_mean_index"][:]; wf = u["waveform_mean"]; wstart = np.concatenate([[0], wf_idx[:-1]])
    W, WE, dist = [], [], []
    for uix in vis:
        w = np.asarray(wf[int(wstart[uix]):int(wf_idx[uix])])
        e0 = ui[uix]
        rows = map_waveform_rows(li, pid, e0)
        if w.ndim != 2 or w.shape[0] != len(rows):
            W.append(np.zeros((82, 32), np.float32)); WE.append(np.zeros((32, 2), np.float32)); dist.append(np.nan); continue
        okr = np.flatnonzero(rows >= 0); er = rows[okr]
        d = np.hypot(hpos[er] - hpos[e0], vpos[er] - vpos[e0])
        near = np.argsort(d, kind="stable")[:32]
        W.append(np.asarray(w[okr[near]], np.float32).T)
        WE.append(np.stack([hpos[er[near]], vpos[er[near]]], axis=1).astype(np.float32))
        tr = int(np.argmin(w[okr].min(axis=1)))
        dist.append(float(np.hypot(hpos[er[tr]] - hpos[e0], vpos[er[tr]] - vpos[e0])))
    metric = {c: u[c][:][vis].astype(float) for c in ("snr", "firing_rate", "isi_violations", "amplitude_cutoff",
                                                       "presence_ratio", "isolation_distance", "d_prime") if c in u}
    unit = {"id": u["id"][:][vis], "area": loc[ui[vis]], "probe": pid[ui[vis]], "vpos": vpos[ui[vis]], "hpos": hpos[ui[vis]],
            "ccf": ccf[ui[vis]], "trough_peak_dist": np.array(dist), **metric}
    np.savez_compressed(OUT / f"allen_{sess}.npz", **{f"unit_{k}": v for k, v in unit.items()},
                        wave=np.stack(W), wave_pos=np.stack(WE), **{f"{k}_counts": c for k, c in counts.items()},
                        **{f"{k}_{c}": v for k, d in stim.items() for c, v in d.items()})
    img = OUT / "allen_ns_images.npy"
    if not img.exists():
        try:
            g = f["stimulus/templates"]
            key = [k for k in g.keys() if "natural_scenes" in k.lower()]
            if key:
                d = g[key[0]]["data"] if "data" in g[key[0]] else g[key[0]]
                np.save(img, np.asarray(d[:, ::4, ::4]).astype(np.uint8))
        except Exception as e:  # noqa: BLE001
            print("no images:", type(e).__name__, str(e)[:80], flush=True)
    return {"status": "ok", "session": sess, "n_vis_good_units": int(len(vis)), "areas": sorted(set(unit["area"].tolist())),
            "trough_peak_dist_median_um": float(np.nanmedian(dist)), "trough_on_peak_frac": float(np.nanmean(np.array(dist) == 0)),
            "trials": {k: int(len(v["start"])) for k, v in stim.items()}}


def process(a):
    import remfile
    url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
    rf = remfile.File(url, _min_chunk_size=4 << 20, _max_threads=8, _max_chunk_size=64 << 20)
    with h5py.File(rf, "r") as f:
        return _process(f, sess_of(a))


def fetch(max_new: int = 6):
    split = json.loads(SPLIT.read_text())
    inv = json.loads(INV.read_text()) if INV.exists() else {}
    by = {sess_of(a): a for a in ASSETS}
    plan = [("dev", s) for s in split["dev"]] + [("select_p13", s) for s in split["select_p13"]]
    for grp in ("train_candidates", "final_candidates"):
        plan += [(grp, s) for s in split[grp]]
    n_ok = {"train_candidates": 0, "final_candidates": 0}
    for grp, s in plan:
        if grp in n_ok and n_ok[grp] >= max_new:
            continue
        if s in inv and inv[s].get("status") == "ok":
            n_ok[grp] = n_ok.get(grp, 0) + 1; continue
        if s in inv and inv[s].get("status", "").startswith("skipped"):
            continue
        t = time.time()
        try:
            r = process(by[s])
        except Exception as e:  # noqa: BLE001
            r = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
        r["group"] = grp; inv[s] = r
        if r["status"] == "ok" and grp in n_ok:
            n_ok[grp] += 1
        print(grp, s, r["status"], r.get("n_vis_good_units"), r.get("trough_peak_dist_median_um"), f"{time.time() - t:.0f}s", flush=True)
        INV.write_text(json.dumps(inv, indent=1, default=str) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        make_split()
    else:
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
