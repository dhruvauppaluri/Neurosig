"""Phase 13: stream Allen Brain Observatory Visual Coding Neuropixels sessions (DANDI 000021) and keep only compact,
analysis-ready arrays (data/allen-venv; nothing larger than the compact cache is written to disk).

Per session (files with drifting gratings, static gratings and natural scenes = Brain Observatory 1.1 sessions):
  units      visual-cortex units (electrode location starting 'VIS', quality 'good'): id, area, probe id, probe
             vertical / horizontal position of the peak channel, CCF x/y/z of the peak channel, snr, firing rate,
             mean waveform on the 32 contacts of the same probe nearest the peak (with their probe positions)
  stimuli    per presentation: start time and labels (DG: orientation, temporal frequency, contrast; SG: orientation,
             spatial frequency, phase; NS: frame), and binned spike counts [trials, units, 10 bins of 25 ms] over
             0-250 ms after onset (int16).
Writes data/phase13/allen_<session>.npz and appends to reports/phase13/allen_inventory.json.
Usage: fetch_allen13.py <n_sessions>
"""
import json, sys, time
from pathlib import Path
import h5py, numpy as np, remfile

ASSETS = json.loads(Path("data/phase13/allen_assets.json").read_text())
BIN_S, NBIN = 0.025, 10
OUT = Path("data/phase13"); INV = Path("reports/phase13/allen_inventory.json")
inv = json.loads(INV.read_text()) if INV.exists() else {}


def strs(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x[:]])


def num(x):
    """Label column to float; 'N/A' / 'null' / blank-sweep markers become NaN."""
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


def table(g, cols):
    return {c: g[c][:] for c in cols if c in g}


def download(url, dest):
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 << 20):
                fh.write(chunk)


def process(a):
    """Download the session file to a temporary path (one at a time, deleted afterwards: local disk is limited), then
    read only the needed tables."""
    url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
    # full-file download measured at ~0.5 MB/s here (2-3 GB per session): stream instead with large, parallel range
    # reads of only the needed datasets (visual units' spike-time blocks, tables, waveforms)
    rf = remfile.File(url, _min_chunk_size=4 << 20, _max_threads=8, _max_chunk_size=64 << 20)
    with h5py.File(rf, "r") as f:
        return _process(f, a)


def _process(f, a):
    iv = f["intervals"]
    if not all(k in iv for k in ("drifting_gratings_presentations", "static_gratings_presentations", "natural_scenes_presentations")):
        return {"status": "skipped: not a Brain Observatory 1.1 session (missing DG/SG/NS)"}
    el = f["general/extracellular_ephys/electrodes"]
    eid = el["id"][:]; loc = strs(el["location"]); vpos = el["probe_vertical_position"][:]; hpos = el["probe_horizontal_position"][:]
    pid = el["probe_id"][:] if "probe_id" in el else np.zeros(len(eid))
    ccf = np.stack([el[c][:] for c in ("x", "y", "z")], axis=1)
    e_ix = {int(e): i for i, e in enumerate(eid)}
    u = f["units"]
    peak = u["peak_channel_id"][:]; q = strs(u["quality"])
    ui = np.array([e_ix.get(int(p), -1) for p in peak])
    vis = np.flatnonzero((ui >= 0) & np.char.startswith(loc[np.maximum(ui, 0)], "VIS") & (q == "good"))
    if len(vis) < 20:
        return {"status": f"skipped: {len(vis)} good visual units"}
    st_index = u["spike_times_index"][:]
    starts = np.concatenate([[0], st_index[:-1]])
    stim = {}
    for key, name, cols in (("dg", "drifting_gratings_presentations", ("orientation", "temporal_frequency", "contrast")),
                            ("sg", "static_gratings_presentations", ("orientation", "spatial_frequency", "phase")),
                            ("ns", "natural_scenes_presentations", ("frame",))):
        g = iv[name]
        stim[key] = {"start": g["start_time"][:], **{c: num(g[c]) for c in cols}}
    edges = np.arange(NBIN + 1) * BIN_S
    counts = {k: np.zeros((len(v["start"]), len(vis), NBIN), np.int16) for k, v in stim.items()}
    t_lo = min(v["start"].min() for v in stim.values()) - 1; t_hi = max(v["start"].max() for v in stim.values()) + 1
    st_ds = u["spike_times"]
    # read contiguous blocks of consecutive visual units in one request each
    blocks, cur = [], [vis[0]]
    for x in vis[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            blocks.append(cur); cur = [x]
    blocks.append(cur)
    spikes = {}
    for b in blocks:
        lo, hi = int(starts[b[0]]), int(st_index[b[-1]])
        arr = st_ds[lo:hi]
        for x in b:
            spikes[int(x)] = arr[int(starts[x]) - lo:int(st_index[x]) - lo]
    for j, uix in enumerate(vis):
        s = spikes[int(uix)]
        s = s[(s >= t_lo) & (s <= t_hi)]
        for k, v in stim.items():
            on = v["start"]
            a0 = np.searchsorted(s, on); a1 = np.searchsorted(s, on + NBIN * BIN_S)
            for t in np.flatnonzero(a1 > a0):
                counts[k][t, j] = np.histogram(s[a0[t]:a1[t]] - on[t], edges)[0]
    # mean waveforms on the 32 nearest contacts of the same probe
    wf_idx = u["waveform_mean_index"][:]; wf = u["waveform_mean"]
    wstart = np.concatenate([[0], wf_idx[:-1]])
    W, WE = [], []
    for uix in vis:
        w = wf[int(wstart[uix]):int(wf_idx[uix])]
        e0 = ui[uix]
        same = np.flatnonzero(pid == pid[e0])
        n_ch = w.shape[0] if w.ndim == 2 else 0
        if n_ch != len(same):
            W.append(np.zeros((82, 32), np.float32)); WE.append(np.zeros((32, 2), np.float32)); continue
        d = np.hypot(hpos[same] - hpos[e0], vpos[same] - vpos[e0])
        near = np.argsort(d)[:32]
        W.append(np.asarray(w[near], np.float32).T); WE.append(np.stack([hpos[same][near], vpos[same][near]], axis=1).astype(np.float32))
    unit = {"id": u["id"][:][vis], "area": loc[ui[vis]], "probe": pid[ui[vis]], "vpos": vpos[ui[vis]], "hpos": hpos[ui[vis]],
            "ccf": ccf[ui[vis]], "snr": u["snr"][:][vis], "rate": u["firing_rate"][:][vis]}
    sess = a["path"].split("ses-")[1].split(".")[0]
    np.savez_compressed(OUT / f"allen_{sess}.npz", **{f"unit_{k}": v for k, v in unit.items()},
                        wave=np.stack(W), wave_pos=np.stack(WE),
                        **{f"{k}_counts": c for k, c in counts.items()},
                        **{f"{k}_{c}": v for k, d in stim.items() for c, v in d.items()})
    return {"status": "ok", "session": sess, "subject": a["path"].split("/")[0], "n_vis_good_units": int(len(vis)),
            "areas": sorted(set(unit["area"].tolist())), "n_probes_vis": int(len(set(unit["probe"].tolist()))),
            "trials": {k: int(len(v["start"])) for k, v in stim.items()},
            "labels": {"dg_orientation": sorted(set(np.round(stim["dg"]["orientation"][np.isfinite(stim["dg"]["orientation"])], 1).tolist())),
                       "sg_orientation": sorted(set(np.round(stim["sg"]["orientation"][np.isfinite(stim["sg"]["orientation"])], 1).tolist())),
                       "ns_frames": int(len(set(stim["ns"]["frame"].tolist())))}}


n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 8
done = sum(1 for v in inv.values() if v.get("status") == "ok")
for a in sorted(ASSETS, key=lambda x: x["path"]):
    if done >= n_target:
        break
    if a["path"] in inv:
        continue
    t = time.time()
    try:
        inv[a["path"]] = process(a)
    except Exception as e:  # noqa: BLE001
        inv[a["path"]] = {"status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
    done += inv[a["path"]].get("status") == "ok"
    print(a["path"], inv[a["path"]].get("status"), inv[a["path"]].get("n_vis_good_units"), f"{time.time() - t:.0f}s", flush=True)
    INV.write_text(json.dumps(inv, indent=1, default=str) + "\n")
